"""Equivalence and resumability tests for segmented Monte Carlo sampling.

Run with ``PYTHONPATH=. pytest tests`` from ``backend``. The tests avoid any
network or container dependency: they exercise the pure ``domain`` layer and the
``workflow``/worker persistence layer against a temporary SQLite database.
"""

import os
import random
import time
from datetime import datetime, timedelta
from fractions import Fraction

import pytest
from sqlalchemy import func, insert, select, update

os.environ.setdefault('DATABASE_URL', 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'test.db'))
os.environ.setdefault('ARTIFACTS_DIR', os.path.join(os.path.dirname(__file__), 'artifacts'))

from src.constants.enums import State  # noqa: E402
from src.models.database import engine  # noqa: E402
from src.models.tables import datasets, metadata, runs, sampling_segments  # noqa: E402
from src.services import domain, workflow  # noqa: E402


MONTE_CARLO_OPTIONS = {'washout_hours': '2', 'min_calls': 3, 'coverage_pct': 50,
                       'min_gain_pp': 0, 'alpha': 0.05, 'seed': 1}


def large_rows(pairs=10, zones=2, seed=7):
    """A ledger whose assignment space exceeds 50,000, forcing Monte Carlo."""
    rnd = random.Random(seed)
    start = datetime(2026, 11, 9, 8)
    rows = []
    for z in range(zones):
        orders = ['AB'] * (pairs // 2) + ['BA'] * (pairs - pairs // 2)
        rnd.shuffle(orders)
        for pair, order in enumerate(orders, 1):
            for block, guide in enumerate(order, 1):
                block_start = start + timedelta(hours=24 * (pair - 1) + 12 * (block - 1))
                for slot in range(6):
                    eligible = 5 + (pair + slot + block + z) % 3
                    on_time = max(0, eligible - (1 if (slot + z + block) % 3 == 0 else 0))
                    rows.append({
                        'zone': f'区{z}', 'planned_zones': zones,
                        'plan_start': start.strftime('%Y-%m-%d %H:%M'),
                        'planned_pairs': pairs, 'pair': pair, 'block': block,
                        'window_start': (block_start + timedelta(hours=2 * slot)).strftime('%Y-%m-%d %H:%M'),
                        'sequence': order, 'guide': guide, 'eligible': eligible, 'on_time': on_time,
                    })
    return rows


def sampling_plan(rows, options=MONTE_CARLO_OPTIONS):
    prepared = domain.prepare(rows, options)
    assert prepared.plan is not None, 'test data must exceed the exact-enumeration limit'
    return prepared.plan


def test_segmented_draws_match_whole_round_including_permutations():
    strata = [[('AB', Fraction(3, 7)), ('BA', Fraction(-2, 5)), ('BA', Fraction(1, 3)),
               ('AB', Fraction(5, 11)), ('BA', Fraction(1, 2)), ('AB', Fraction(-1, 9))],
              [('BA', Fraction(2, 3)), ('AB', Fraction(1, 6)), ('BA', Fraction(-4, 7)),
               ('AB', Fraction(3, 8)), ('BA', Fraction(1, 2)), ('AB', Fraction(-3, 10))]]
    plan = domain.build_sampling_plan(strata, seed=691124)
    plan = domain.SamplingPlan(plan.scores, plan.observed, plan.seed, draws=30_000, chunk=4_000)
    assert sum(domain.sample_segment(plan, i) for i in range(plan.segments)) == domain.whole_round_hits(plan)


@pytest.mark.parametrize('seed', [691124, 1, 2**31 - 1])
def test_monte_carlo_metrics_identical(seed):
    rows = large_rows()
    options = dict(MONTE_CARLO_OPTIONS, seed=seed)
    prepared = domain.prepare(rows, options)
    assert prepared.plan is not None
    stored = {i: domain.sample_segment(prepared.plan, i) for i in range(prepared.plan.segments)}
    segmented = domain.finalize(prepared, stored)
    whole = domain.compute(rows, options)
    assert segmented == whole


def test_out_of_order_and_duplicate_segments_count_once():
    rows = large_rows()
    plan = sampling_plan(rows)
    order = list(range(plan.segments))
    random.Random(3).shuffle(order)
    hits = {}
    for index in order:
        hits.setdefault(index, domain.sample_segment(plan, index))  # duplicates are ignored client-side too
        hits.setdefault(index, domain.sample_segment(plan, index))
    assert len(hits) == plan.segments
    combined, draws, counted, method = domain.combine_segments(plan, hits)
    assert (draws, counted, method) == (plan.draws, domain.whole_round_hits(plan), '蒙特卡洛抽样')


@pytest.fixture
def database():
    metadata.drop_all(engine)
    metadata.create_all(engine)
    yield engine
    metadata.drop_all(engine)


@pytest.fixture
def queued_run(database):
    rows = large_rows()
    with database.begin() as connection:
        dataset = connection.execute(insert(datasets).values(
            name='t', checksum='c' * 64, records=rows, owner='a', created=time.time())).inserted_primary_key[0]
        run = connection.execute(insert(runs).values(
            dataset_id=dataset, options=MONTE_CARLO_OPTIONS, state=State.queued.value, owner='a',
            created=time.time())).inserted_primary_key[0]
    return run


def test_worker_resumes_after_interruption(queued_run, database):
    import src.worker as worker

    job = workflow.claim_run()
    assert job['id'] == queued_run
    prepared = domain.prepare(job['rows'], job['options'])
    plan = prepared.plan

    # Simulate a stop after drawing a prefix of segments.
    original = worker._sample

    def stop_halfway(job, prepared):
        key, fingerprint = job['id'], prepared.plan.fingerprint()
        workflow.begin_sampling(key, job['claim'], fingerprint, prepared.plan.segments)
        rng = domain.open_segment(prepared.plan, 0)
        for index in range(prepared.plan.segments // 2 + 1):
            hit = domain.draw_segment(prepared.plan, index, rng)
            workflow.save_segment(key, fingerprint, index, hit, domain.dump_state(rng))
        return None

    worker._sample = stop_halfway
    try:
        worker.calculate(job)
    finally:
        worker._sample = original

    with database.begin() as connection:
        row = connection.execute(select(runs).where(runs.c.id == queued_run)).mappings().one()
    assert row['state'] == State.running.value and row['result'] is None

    # Simulate the claim expiring so a fresh worker can take over and finish.
    with database.begin() as connection:
        connection.execute(update(runs).where(runs.c.id == queued_run).values(started=time.time() - 200))
    worker.calculate(workflow.claim_run())

    with database.begin() as connection:
        row = connection.execute(select(runs).where(runs.c.id == queued_run)).mappings().one()
        stored = connection.scalar(select(func.count()).select_from(sampling_segments)
                                   .where(sampling_segments.c.run_id == queued_run))
    assert row['state'] == State.completed.value
    assert stored == plan.segments
    expected = domain.compute(job['rows'], job['options'])
    for key in ('headline', 'metrics', 'rows', 'notes', 'chart', 'can_publish'):
        assert row['result'][key] == expected[key]


def test_takeover_accepts_stale_segments_once(queued_run, database):
    first = workflow.claim_run()
    prepared = domain.prepare(first['rows'], first['options'])
    plan = prepared.plan
    fingerprint = plan.fingerprint()
    workflow.begin_sampling(first['id'], first['claim'], fingerprint, plan.segments)

    rng = domain.open_segment(plan, 0)
    for index in range(5):
        workflow.save_segment(first['id'], fingerprint, index, domain.draw_segment(plan, index, rng),
                              domain.dump_state(rng))

    with database.begin() as connection:
        connection.execute(update(runs).where(runs.c.id == queued_run).values(started=time.time() - 200))
    second = workflow.claim_run()
    assert second['claim'] != first['claim']
    workflow.begin_sampling(second['id'], second['claim'], fingerprint, plan.segments)

    # The stale worker keeps submitting without a live claim; its segments must
    # still be accepted (same round) and never double-counted.
    for index in range(5, 8):
        assert workflow.save_segment(first['id'], fingerprint, index, domain.sample_segment(plan, index), 'x')
        assert workflow.save_segment(second['id'], fingerprint, index, domain.sample_segment(plan, index), 'x')

    with database.begin() as connection:
        stored = connection.scalar(select(func.count()).select_from(sampling_segments)
                                   .where(sampling_segments.c.run_id == queued_run))
    assert stored == 8

    # A segment from a different round is rejected.
    assert workflow.save_segment(first['id'], 'deadbeef', 0, 0, 'x') is False
