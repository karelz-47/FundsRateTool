from sqlalchemy.dialects.postgresql import insert as pg_insert
from db import PublishedRate

def upsert_published_rates(session, df_long, source="xlsm_backfill", chunk_size=5000):
    if df_long.empty:
        return 0

    payload = df_long.copy()
    payload["source"] = source
    total = 0
    rows = payload.to_dict("records")

    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i+chunk_size]
        stmt = pg_insert(PublishedRate).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=["rate_date", "series_code"],
            set_={
                "value": stmt.excluded.value,
                "source": stmt.excluded.source,
            },
        )
        session.execute(stmt)
        total += len(chunk)

    session.commit()
    return total
