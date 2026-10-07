# ==============================================================
# File Name : database.py
# Project   : Crop Yield Prediction System
# Description:
# Connects FastAPI to PostgreSQL using SQLAlchemy.
# Every route that needs the database uses get_db() dependency.
# ==============================================================

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ==============================================================
# Step 1 — Database connection settings
# Change DB_PASSWORD to match your PostgreSQL password
# ==============================================================
DB_USERNAME = "postgres"
DB_PASSWORD = "pankajpostgres"   # ← change this if different
DB_HOST     = "localhost"
DB_PORT     = "5432"
DB_NAME     = "crop_yield_db"

# ==============================================================
# Step 2 — Build the connection URL
# Format: postgresql://user:password@host:port/database
# ==============================================================
DATABASE_URL = "postgresql://postgres:pankajpostgres@localhost:5432/crop_yield_db"

# ==============================================================
# Step 3 — Create Engine
# Engine manages the actual connection pool to PostgreSQL.
# echo=True → prints every SQL query in terminal (for debugging)
# Set echo=False in production to reduce log noise
# ==============================================================
engine = create_engine(DATABASE_URL, echo=True)

# ==============================================================
# Step 4 — Session Factory
# Creates one database session per request.
# autocommit=False → we call db.commit() manually
# autoflush=False  → we control when changes go to DB
# ==============================================================
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ==============================================================
# Step 5 — Base Class
# All models (tables) inherit from Base.
# Base.metadata.create_all(engine) creates all tables.
# ==============================================================
Base = declarative_base()

# ==============================================================
# Step 6 — get_db() Dependency
# Used in every route that needs database access.
# FastAPI calls this automatically via Depends(get_db).
#
# Flow:
#   1. Open a new session for the request
#   2. Give session to the route function
#   3. After route finishes, close session (even on error)
#
# Usage:
#   def my_route(db: Session = Depends(get_db)):
# ==============================================================
def get_db():
    db = SessionLocal()
    try:
        yield db       # give DB session to the route
    finally:
        db.close()     # always close, even if error occurs
