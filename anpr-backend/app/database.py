import os

from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

_BASE_DIR = Path(__file__).resolve().parent.parent

def resolve_database_url():
    """Read DATABASE_URL from the environment or anpr-backend/.env.

    A real environment variable wins over the .env file. There is no
    built-in username or password: startup stops if this is missing.
    """
    load_dotenv(_BASE_DIR / ".env", override=False)
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. In the anpr-backend folder, copy "
            ".env.example to .env and set DATABASE_URL to your PostgreSQL "
            "database. Local setup creates that database in SETUP.md. "
            "The app will not guess a password."
        )
    # SQLAlchemy 2 treats plain postgresql:// as the psycopg v3 driver,
    # which this project does not install.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url

DATABASE_URL = resolve_database_url()

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

class Base(DeclarativeBase):
    pass
