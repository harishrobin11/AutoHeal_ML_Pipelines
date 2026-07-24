import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from config.settings import settings

class TelemetryIngestor:
    """Handles persistence of telemetry stream logs into SQLite or PostgreSQL database."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes database schema from schema.sql if tables don't exist."""
        schema_path = os.path.join(os.path.dirname(__file__), "../../database/schema.sql")
        schema_path = os.path.abspath(schema_path)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if os.path.exists(schema_path):
                with open(schema_path, "r") as f:
                    cursor.executescript(f.read())
            else:
                # Fallback schema definition if file not found
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id VARCHAR(36) PRIMARY KEY,
                    service_name VARCHAR(100) NOT NULL,
                    endpoint VARCHAR(255) NOT NULL,
                    status_code INTEGER NOT NULL,
                    response_time_ms DOUBLE PRECISION NOT NULL,
                    payload_json TEXT,
                    payload_schema_hash VARCHAR(64),
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
            conn.commit()

    def ingest_records(self, records: List[Dict[str, Any]]) -> int:
        """Inserts a list of telemetry log dictionaries into the database."""
        query = """
        INSERT INTO telemetry_logs (
            id, service_name, endpoint, status_code, response_time_ms, payload_json, payload_schema_hash, error_message, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = [
            (
                r["id"],
                r["service_name"],
                r["endpoint"],
                r["status_code"],
                r["response_time_ms"],
                r.get("payload_json"),
                r.get("payload_schema_hash"),
                r.get("error_message"),
                r["timestamp"]
            )
            for r in records
        ]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, data)
            conn.commit()
            return cursor.rowcount

    def fetch_all(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetches latest telemetry logs."""
        query = "SELECT * FROM telemetry_logs ORDER BY timestamp DESC LIMIT ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
