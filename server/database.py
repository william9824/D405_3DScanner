"""psycopg3 connection pool. Opened/closed by the FastAPI lifespan."""
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import DATABASE_URL

# open=False: pool 喺 FastAPI lifespan 先正式 open，避免 import 時就連線
pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=5,
    kwargs={"row_factory": dict_row},
    open=False,
)
