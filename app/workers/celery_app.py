"""Celery application factory."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "flokka",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.workers.ingestion_worker",
        "app.workers.embedding_worker",
        "app.workers.indexing_worker",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={
        "socket_connect_timeout": 2,
        "socket_timeout": 2,
        "retry_on_timeout": False,
        "max_retries": 1,
    },
    broker_connection_retry_on_startup=False,
)
