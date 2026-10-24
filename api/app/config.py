from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    inference_url: str = "http://inference:8001"
    log_level: str = "INFO"
    videos_dir: str = "/app/videos"
    frontend_dir: str = "/app/frontend"
    max_workers: int = 4
    target_width: int = 640
    jpeg_quality_inference: int = 80
    jpeg_quality_stream: int = 80
    score_threshold: float = 0.6
    track_iou_threshold: float = 0.3
    track_max_missed: int = 15
    track_min_hits: int = 3
    stream_timeout: float = 10.0


settings = Settings()
