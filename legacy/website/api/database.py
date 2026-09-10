import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get DB URL from env or fallback to local sqlite
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////home/ubuntu/neuroshard/website/api/users.db")

if SQLALCHEMY_DATABASE_URL.startswith("postgres"):
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
else:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
