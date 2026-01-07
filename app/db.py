# app/db.py
from sqlmodel import create_engine, Session
from sqlalchemy import event

SQLITE_FILE = "app.db"
DATABASE_URL = f"sqlite:///{SQLITE_FILE}"

# check_same_thread=False is needed for typical FastAPI usage with SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()

def get_session():
    with Session(engine) as session:
        yield session