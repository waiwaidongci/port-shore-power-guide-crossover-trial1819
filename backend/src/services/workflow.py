import time

from fastapi import HTTPException
from sqlalchemy import and_, delete, insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from src.constants.enums import State
from src.models.database import add_event, engine, fetch_one
from src.models.tables import datasets, deliveries, runs, sampling_segments, schedules
from src.services.data import parse_options


def add_run(body, actor):
    options = parse_options(body.options)
    now = time.time()
    with engine.begin() as connection:
        if fetch_one(connection, datasets, body.dataset_id) is None:
            raise HTTPException(404, '数据版本不存在。')
        key = connection.execute(insert(runs).values(dataset_id=body.dataset_id, options=options,
                    state=State.queued.value, owner=actor, created=now)).inserted_primary_key[0]
        add_event(connection, actor, 'queue', key, now)
    return {'id': key, 'state': State.queued.value}


def claim_run():
    now = time.time()
    claimable = or_(runs.c.state == State.queued.value, and_(runs.c.state == State.running.value, runs.c.started < now - 120))
    with engine.begin() as connection:
        run = connection.execute(select(runs).where(claimable).order_by(runs.c.id).limit(1)).mappings().one_or_none()
        if run is None:
            return None
        key = run['id']
        claim = connection.execute(update(runs).where(runs.c.id == key, claimable)
                                   .values(state=State.running.value, started=now))
        if claim.rowcount != 1:
            return None
        data = fetch_one(connection, datasets, run['dataset_id'])
        return {'id': key, 'claim': now, 'rows': data['records'], 'options': run['options']}


def finish_run(key, claim, result, error):
    now = time.time()
    state = State.failed.value if error else State.completed.value
    with engine.begin() as connection:
        saved = connection.execute(update(runs).where(runs.c.id == key, runs.c.started == claim,
                                   runs.c.state == State.running.value).values(state=state, result=result,
                                                  error=error, finished=now))
        if saved.rowcount:
            add_event(connection, 'worker', state, key, now)


def save_segment(key, fingerprint, index, hits, rng_state):
    """Persist one drawn segment exactly once, tolerating out-of-order delivery.

    Segments are identified by ``(run_id, fingerprint, segment)``; the unique
    constraint makes a duplicate insert fail and be ignored, so each segment
    counts exactly once no matter how many workers submit it or in what order.
    Acceptance depends on the round, not the submitter's claim, so a worker
    that was taken over mid-round can still contribute its segments. Segments
    from a superseded round (different fingerprint) or a run that already
    finished are rejected.
    """
    now = time.time()
    with engine.begin() as connection:
        run = fetch_one(connection, runs, key)
        if run is None or run['state'] != State.running.value or run['sampling_fingerprint'] != fingerprint:
            return False
        try:
            with connection.begin_nested():
                connection.execute(insert(sampling_segments).values(
                    run_id=key, fingerprint=fingerprint, segment=index, hits=hits,
                    rng_state=rng_state, at=now))
        except IntegrityError:
            pass
    return True


def begin_sampling(key, claim, fingerprint, total):
    """Open a new round, discarding any stored segments left from a prior round."""
    with engine.begin() as connection:
        run = fetch_one(connection, runs, key)
        if run is None or run['state'] != State.running.value or run['started'] != claim:
            return False
        if run['sampling_fingerprint'] != fingerprint:
            connection.execute(delete(sampling_segments).where(sampling_segments.c.run_id == key))
            connection.execute(update(runs).where(runs.c.id == key).values(sampling_fingerprint=fingerprint,
                               sampling_total=total))
    return True


def load_segment_hits(key, fingerprint):
    with engine.connect() as connection:
        rows = connection.execute(select(sampling_segments.c.segment, sampling_segments.c.hits)
                                  .where(sampling_segments.c.run_id == key,
                                         sampling_segments.c.fingerprint == fingerprint)).all()
    return {segment: hits for segment, hits in rows}


def load_segment_state(key, fingerprint, index):
    """Return the stored RNG position *after* a segment, or ``None``."""
    with engine.connect() as connection:
        return connection.scalar(select(sampling_segments.c.rng_state)
                                 .where(sampling_segments.c.run_id == key,
                                        sampling_segments.c.fingerprint == fingerprint,
                                        sampling_segments.c.segment == index))


def review_run(key, body, actor):
    now = time.time()
    note = body.note.strip()
    if len(note) < 2:
        raise ValueError('请输入至少两个字符的复核意见。')
    with engine.begin() as connection:
        run = fetch_one(connection, runs, key)
        if run is None:
            raise HTTPException(404, '分析任务不存在。')
        if run['state'] != State.completed.value:
            raise HTTPException(409, '分析完成后才能复核。')
        if body.decision == 'approved' and run['result'].get('can_publish') is False:
            raise HTTPException(409, '结果未达到本项目的发布门槛。')
        updated = connection.execute(update(runs).where(runs.c.id == key, runs.c.decision.is_(None))
                                     .values(decision=body.decision, reviewer=actor, review_note=note, reviewed=now))
        if updated.rowcount != 1:
            raise HTTPException(409, '已完成复核。请重新运行分析以提交新的复核。')
        add_event(connection, actor, body.decision, key, now)
    return {'decision': body.decision}


def schedule_report(key, body, actor):
    now = time.time()
    recipient = body.recipient.strip()
    if len(recipient) < 2:
        raise ValueError('请填写站内收件组。')
    if body.scheduled_at and body.scheduled_at.utcoffset() is None:
        raise ValueError('投递时间须含时区。')
    timestamp = body.scheduled_at.timestamp() if body.scheduled_at else now
    if body.scheduled_at and timestamp < now - 60:
        raise ValueError('投递时间已过，请调整时间。')
    with engine.begin() as connection:
        run = fetch_one(connection, runs, key)
        if run is None or run['decision'] != 'approved':
            raise HTTPException(409, '仅已批准的结果可分发。')
        required_recipient = run['result'].get('dispatch_recipient')
        if required_recipient and recipient != required_recipient:
            raise HTTPException(422, f'该报告应投递给{required_recipient}。')
        schedule_id = connection.execute(insert(schedules).values(run_id=key, recipient=recipient, cadence=body.cadence,
                                      next_at=timestamp, active=1, owner=actor)).inserted_primary_key[0]
        add_event(connection, actor, 'schedule', schedule_id, now)
    return {'id': schedule_id, 'next_at': timestamp}


def cancel_schedule(key, actor):
    with engine.begin() as connection:
        updated = connection.execute(update(schedules).where(schedules.c.id == key, schedules.c.active == 1)
                                     .values(active=0))
        if not updated.rowcount:
            raise HTTPException(409, '该计划已停止或已完成。')
        add_event(connection, actor, 'cancel', key, time.time())
    return {'id': key, 'active': False}


def publish_due():
    now = time.time()
    with engine.begin() as connection:
        pending = connection.execute(select(schedules).where(schedules.c.active == 1, schedules.c.next_at <= now)
                                     .order_by(schedules.c.next_at, schedules.c.id).limit(20)).mappings().all()
        for item in pending:
            previous = item['next_at']
            daily = item['cadence'] == 'daily'
            future = previous + (int((now - previous) / 86400) + 1) * 86400 if daily else previous
            changed = connection.execute(update(schedules).where(schedules.c.id == item['id'], schedules.c.active == 1,
                              schedules.c.next_at == previous).values(active=int(daily), next_at=future))
            if changed.rowcount:
                delivery_id = connection.execute(insert(deliveries).values(schedule_id=item['id'], run_id=item['run_id'],
                                               recipient=item['recipient'], scheduled=previous, delivered=now)).inserted_primary_key[0]
                add_event(connection, 'worker', 'delivered', delivery_id, now)
