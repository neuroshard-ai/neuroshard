"""
Database Migration Helper for NeuroShard Website API

This module handles schema migrations for adding new tables and columns.
Supports both SQLite (development) and PostgreSQL (production).

Migrations run automatically on app startup via the startup_event in main.py.
You can also run manually:
    python -m api.migrations
"""

import logging
import os
from datetime import datetime, timedelta

from sqlalchemy import text, inspect
from sqlalchemy.exc import OperationalError, ProgrammingError

logger = logging.getLogger(__name__)


def get_engine():
    """Get SQLAlchemy engine from database module."""
    from .database import engine
    return engine


def column_exists(inspector, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    try:
        columns = [col['name'] for col in inspector.get_columns(table)]
        return column in columns
    except Exception:
        return False


def table_exists(inspector, table: str) -> bool:
    """Check if a table exists."""
    return table in inspector.get_table_names()


def migrate_users_table(conn, inspector, is_postgres: bool):
    """Add new columns to users table for chat tracking."""
    logger.info("Migrating users table...")
    
    migrations = [
        ("chat_count", "INTEGER DEFAULT 0"),
        ("total_tokens_used", "INTEGER DEFAULT 0"),
        ("total_neuro_spent_chat", "REAL DEFAULT 0.0" if not is_postgres else "DOUBLE PRECISION DEFAULT 0.0"),
        ("last_chat_at", "TIMESTAMP" if is_postgres else "DATETIME"),
        ("rate_limit_tier", "VARCHAR(50) DEFAULT 'standard'"),
        ("is_rate_limited", "BOOLEAN DEFAULT FALSE"),
        ("rate_limit_until", "TIMESTAMP" if is_postgres else "DATETIME"),
    ]
    
    for column_name, column_def in migrations:
        if not column_exists(inspector, "users", column_name):
            logger.info(f"  Adding column: {column_name}")
            try:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {column_def}"))
                conn.commit()
            except (OperationalError, ProgrammingError) as e:
                # Column might already exist (race condition or partial migration)
                if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                    logger.info(f"  Column already exists: {column_name}")
                else:
                    raise
        else:
            logger.info(f"  Column already exists: {column_name}")


def create_chat_interactions_table(conn, inspector, is_postgres: bool):
    """Create chat_interactions table for tracking."""
    logger.info("Creating chat_interactions table...")
    
    if table_exists(inspector, "chat_interactions"):
        logger.info("  Table already exists: chat_interactions")
        return
    
    # Use appropriate syntax for PostgreSQL vs SQLite
    if is_postgres:
        conn.execute(text("""
            CREATE TABLE chat_interactions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                prompt_length INTEGER NOT NULL,
                max_tokens_requested INTEGER DEFAULT 50,
                response_length INTEGER,
                tokens_used INTEGER,
                neuro_cost DOUBLE PRECISION,
                fee_burned DOUBLE PRECISION,
                response_time_ms INTEGER,
                node_response_time_ms INTEGER,
                success BOOLEAN DEFAULT TRUE,
                error_code VARCHAR(50),
                error_message TEXT,
                target_node_url VARCHAR(500),
                nodes_tried INTEGER DEFAULT 1,
                client_ip VARCHAR(64),
                user_agent VARCHAR(500)
            )
        """))
    else:
        conn.execute(text("""
            CREATE TABLE chat_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                prompt_length INTEGER NOT NULL,
                max_tokens_requested INTEGER DEFAULT 50,
                response_length INTEGER,
                tokens_used INTEGER,
                neuro_cost REAL,
                fee_burned REAL,
                response_time_ms INTEGER,
                node_response_time_ms INTEGER,
                success BOOLEAN DEFAULT 1,
                error_code VARCHAR,
                error_message TEXT,
                target_node_url VARCHAR,
                nodes_tried INTEGER DEFAULT 1,
                client_ip VARCHAR,
                user_agent VARCHAR,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """))
    
    conn.commit()
    
    # Create indexes
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_chat_interactions_user_id 
        ON chat_interactions (user_id)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_chat_interactions_created_at 
        ON chat_interactions (created_at)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_chat_user_created 
        ON chat_interactions (user_id, created_at)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_chat_created_success 
        ON chat_interactions (created_at, success)
    """))
    conn.commit()
    
    logger.info("  Created table: chat_interactions with indexes")


def create_rate_limit_events_table(conn, inspector, is_postgres: bool):
    """Create rate_limit_events table for monitoring."""
    logger.info("Creating rate_limit_events table...")
    
    if table_exists(inspector, "rate_limit_events"):
        logger.info("  Table already exists: rate_limit_events")
        return
    
    if is_postgres:
        conn.execute(text("""
            CREATE TABLE rate_limit_events (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                client_ip VARCHAR(64) NOT NULL,
                endpoint VARCHAR(200) NOT NULL,
                limit_type VARCHAR(50) NOT NULL,
                limit_value VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                request_count INTEGER,
                user_agent VARCHAR(500)
            )
        """))
    else:
        conn.execute(text("""
            CREATE TABLE rate_limit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                client_ip VARCHAR NOT NULL,
                endpoint VARCHAR NOT NULL,
                limit_type VARCHAR NOT NULL,
                limit_value VARCHAR NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                request_count INTEGER,
                user_agent VARCHAR,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """))
    
    conn.commit()
    
    # Create indexes
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_rate_limit_events_client_ip 
        ON rate_limit_events (client_ip)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_rate_limit_events_created_at 
        ON rate_limit_events (created_at)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_ratelimit_ip_created 
        ON rate_limit_events (client_ip, created_at)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_ratelimit_user_created 
        ON rate_limit_events (user_id, created_at)
    """))
    conn.commit()
    
    logger.info("  Created table: rate_limit_events with indexes")


def migrate():
    """Run all migrations."""
    engine = get_engine()
    database_url = str(engine.url)
    is_postgres = "postgresql" in database_url.lower()
    
    logger.info(f"Running migrations on database: {'PostgreSQL' if is_postgres else 'SQLite'}")
    
    try:
        with engine.connect() as conn:
            inspector = inspect(engine)
            
            # Check if users table exists (required)
            if not table_exists(inspector, "users"):
                logger.warning("Users table doesn't exist yet. Skipping migrations.")
                logger.info("Migrations will be applied after tables are created by SQLAlchemy.")
                return False
            
            # Run migrations
            migrate_users_table(conn, inspector, is_postgres)
            create_chat_interactions_table(conn, inspector, is_postgres)
            create_rate_limit_events_table(conn, inspector, is_postgres)
        
        logger.info("All migrations completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        # Don't raise - allow app to start even if migrations fail
        # Tables will be created by SQLAlchemy models
        return False


def verify_schema():
    """Verify that all expected tables and columns exist."""
    logger.info("Verifying schema...")
    
    engine = get_engine()
    
    results = {
        "status": "ok",
        "database": "postgresql" if "postgresql" in str(engine.url) else "sqlite",
        "tables": {},
        "issues": []
    }
    
    try:
        inspector = inspect(engine)
        
        # Check users table
        user_columns = [
            "chat_count", "total_tokens_used", "total_neuro_spent_chat",
            "last_chat_at", "rate_limit_tier", "is_rate_limited", "rate_limit_until"
        ]
        
        if table_exists(inspector, "users"):
            results["tables"]["users"] = {"exists": True, "columns": {}}
            for col in user_columns:
                exists = column_exists(inspector, "users", col)
                results["tables"]["users"]["columns"][col] = exists
                if not exists:
                    results["issues"].append(f"Missing column: users.{col}")
        else:
            results["tables"]["users"] = {"exists": False}
            results["issues"].append("Missing table: users")
        
        # Check chat_interactions table
        if table_exists(inspector, "chat_interactions"):
            results["tables"]["chat_interactions"] = {"exists": True}
        else:
            results["tables"]["chat_interactions"] = {"exists": False}
            results["issues"].append("Missing table: chat_interactions")
        
        # Check rate_limit_events table
        if table_exists(inspector, "rate_limit_events"):
            results["tables"]["rate_limit_events"] = {"exists": True}
        else:
            results["tables"]["rate_limit_events"] = {"exists": False}
            results["issues"].append("Missing table: rate_limit_events")
        
        if results["issues"]:
            results["status"] = "needs_migration"
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results


def cleanup_old_data(days: int = 30):
    """
    Clean up old tracking data to prevent database bloat.
    
    Args:
        days: Delete data older than this many days
    """
    logger.info(f"Cleaning up data older than {days} days...")
    
    engine = get_engine()
    is_postgres = "postgresql" in str(engine.url)
    
    try:
        with engine.connect() as conn:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Delete old chat interactions
            if is_postgres:
                result = conn.execute(text("""
                    DELETE FROM chat_interactions 
                    WHERE created_at < :cutoff
                """), {"cutoff": cutoff})
            else:
                result = conn.execute(text("""
                    DELETE FROM chat_interactions 
                    WHERE created_at < :cutoff
                """), {"cutoff": cutoff.isoformat()})
            chat_deleted = result.rowcount
            
            # Delete old rate limit events
            if is_postgres:
                result = conn.execute(text("""
                    DELETE FROM rate_limit_events 
                    WHERE created_at < :cutoff
                """), {"cutoff": cutoff})
            else:
                result = conn.execute(text("""
                    DELETE FROM rate_limit_events 
                    WHERE created_at < :cutoff
                """), {"cutoff": cutoff.isoformat()})
            rate_deleted = result.rowcount
            
            conn.commit()
        
        logger.info(f"Deleted {chat_deleted} chat interactions, {rate_deleted} rate limit events")
        
        return {
            "status": "ok",
            "deleted": {
                "chat_interactions": chat_deleted,
                "rate_limit_events": rate_deleted,
            }
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run migrations
    print("=" * 60)
    print("NeuroShard Website API - Database Migration")
    print("=" * 60)
    
    # First verify current state
    print("\nVerifying current schema...")
    status = verify_schema()
    print(f"Database: {status.get('database', 'unknown')}")
    print(f"Status: {status['status']}")
    
    if status.get("issues"):
        print("\nIssues found:")
        for issue in status["issues"]:
            print(f"  - {issue}")
        
        print("\nRunning migrations...")
        migrate()
        
        # Verify again
        print("\nVerifying after migration...")
        status = verify_schema()
        print(f"Status: {status['status']}")
        
        if status.get("issues"):
            print("\nRemaining issues:")
            for issue in status["issues"]:
                print(f"  - {issue}")
        else:
            print("\n✅ All migrations applied successfully!")
    else:
        print("\n✅ Schema is up to date, no migrations needed.")
