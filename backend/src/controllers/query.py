from fastapi import HTTPException
from sqlalchemy import func, select

from src.models.database import engine, fetch_one
from src.models.tables import audit, datasets, deliveries, runs, sampling_segments, schedules


def list_rows(table, fields=None, limit=100):
    statement = select(*fields) if fields else select(table)
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(statement.order_by(table.c.id.desc()).limit(limit)).mappings()]


def revisions():
    columns = [field for field in datasets.c if field.name != 'records']
    return list_rows(datasets, columns)


def require_row(table, key, label):
    with engine.connect() as connection:
        row = fetch_one(connection, table, key)
    if row is None:
        raise HTTPException(404, label + '不存在。')
    return dict(row)


def sampling_progress(row):
    """Progress of a resumable Monte Carlo round, or ``None`` for exact/enumerated runs."""
    total = row.get('sampling_total')
    fingerprint = row.get('sampling_fingerprint')
    if not total or not fingerprint:
        return None
    with engine.connect() as connection:
        completed = connection.scalar(select(func.count()).select_from(sampling_segments)
                                      .where(sampling_segments.c.run_id == row['id'],
                                             sampling_segments.c.fingerprint == fingerprint))
    return {'total': total, 'completed': completed, 'ready': completed >= total}


def run_detail(key):
    row = require_row(runs, key, '分析任务')
    progress = sampling_progress(row)
    if progress is not None:
        row['sampling'] = progress
    return row


def jobs():
    return list_rows(runs, [field for field in runs.c if field.name != 'result'])


def delivery_status():
    return {'schedules': list_rows(schedules), 'deliveries': list_rows(deliveries)}


def recent_activity():
    return list_rows(audit, limit=20)
