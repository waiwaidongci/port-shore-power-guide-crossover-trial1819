import json
import logging
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.config.settings import ARTIFACTS
from src.models.database import cache
from src.services.domain import dump_state, draw_segment, finalize, load_state, open_segment, prepare
from src.services.workflow import (begin_sampling, claim_run, finish_run, load_segment_hits,
                                   load_segment_state, publish_due, save_segment)
from src.utils.exports import write_csv


def _artifact(report, key, claim):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=ARTIFACTS, delete=False) as temporary:
        temporary.write(write_csv(report['rows'], report['columns']))
    Path(temporary.name).replace(ARTIFACTS / f'run-{key}-{claim.hex()}.csv')


def _sample(job, prepared):
    """Draw every segment of a Monte Carlo round, resuming stored ones.

    Each claim opens the round once. The first missing segment is the resume
    point: the RNG is restored from the stored position of the segment just
    before it (or replayed from the seed when it is the first), then the
    remaining segments are drawn in order. Segments stored by a prior claim are
    never redrawn, so a worker stopped mid-round only fills what is missing.
    The conclusion is produced only once the full set is stored.
    """
    key, claim = job['id'], job['claim']
    plan = prepared.plan
    fingerprint = plan.fingerprint()
    if not begin_sampling(key, claim, fingerprint, plan.segments):
        return None
    stored = load_segment_hits(key, fingerprint)
    missing = [index for index in range(plan.segments) if index not in stored]
    if missing:
        first = missing[0]
        prior = load_segment_state(key, fingerprint, first - 1) if first else None
        rng = load_state(prior) if prior is not None else open_segment(plan, first)
        for index in range(first, plan.segments):
            hits = draw_segment(plan, index, rng)
            if index in stored:
                continue
            if not save_segment(key, fingerprint, index, hits, dump_state(rng)):
                return None
    complete = load_segment_hits(key, fingerprint)
    return complete if len(complete) == plan.segments else None


def calculate(job):
    key, claim = job['id'], job['claim']
    try:
        prepared = prepare(job['rows'], job['options'])
        segments = _sample(job, prepared) if prepared.plan is not None else None
        if prepared.plan is not None and segments is None:
            return
        report = finalize(prepared, segments)
        report['columns'] = list(dict.fromkeys(row for item in report['rows'] for row in item))
        report['metric_order'] = list(report['metrics'])
        json.dumps(report, allow_nan=False)
        _artifact(report, key, claim)
    except ValueError as error:
        finish_run(key, claim, None, str(error))
    except Exception:
        logging.exception('Analysis %s failed', key)
        finish_run(key, claim, None, '计算失败，请核对数据并重新提交。')
    else:
        finish_run(key, claim, report, None)


def work():
    while True:
        try:
            job = claim_run()
            if job:
                calculate(job)
            publish_due()
            cache.set('worker:heartbeat', str(time.time()), ex=25)
        except Exception:
            logging.exception('Worker iteration failed')
        time.sleep(0.8)


if __name__ == '__main__':
    work()
