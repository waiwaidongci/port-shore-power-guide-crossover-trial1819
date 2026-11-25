"""Idempotent schema upgrades for in-place deployments.

``metadata.create_all`` creates missing tables but never adds columns to a table
that already exists. The resumable-sampling feature adds three columns to
``analysis_runs`` and one table; on an existing database we add the columns here
before the worker starts writing segments.
"""

from sqlalchemy import inspect, text

RUN_COLUMNS = {
    'sampling_fingerprint': 'VARCHAR(64)',
    'sampling_total': 'INTEGER',
}


def _existing(connection, table):
    return {column['name'] for column in inspect(connection).get_columns(table)}


def upgrade(engine):
    inspector = inspect(engine)
    if not inspector.has_table('analysis_runs'):
        return
    present = _existing(engine, 'analysis_runs')
    missing = [(name, ddl) for name, ddl in RUN_COLUMNS.items() if name not in present]
    if not missing:
        return
    with engine.begin() as connection:
        for name, ddl in missing:
            connection.execute(text(f'ALTER TABLE analysis_runs ADD COLUMN {name} {ddl}'))
