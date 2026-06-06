"""psycopg3 connection pool. Opened/closed by the FastAPI lifespan."""
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import DATABASE_URL

# open=False: pool open after FastAPI startup, not at import time
pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=5,
    kwargs={"row_factory": dict_row},
    open=False,
)
