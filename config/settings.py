"""
Configuration settings for AutoHeal-ML application.
"""
import os

class Settings:
    PROJECT_NAME: str = "AutoHeal-ML"
    VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./autoheal.db")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "autoheal.db")
    
    # Anomaly Thresholds
    ZSCORE_THRESHOLD: float = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
    SLIDING_WINDOW_MINUTES: int = int(os.getenv("SLIDING_WINDOW_MINUTES", "15"))
    MIN_WINDOW_SAMPLES: int = int(os.getenv("MIN_WINDOW_SAMPLES", "10"))
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Server Ports
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))

settings = Settings()
