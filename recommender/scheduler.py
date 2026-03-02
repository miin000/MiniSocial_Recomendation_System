"""
scheduler.py
────────────
Tự động gọi hàm train_and_save() theo lịch định kỳ.
Mặc định mỗi 6 tiếng (cấu hình qua RETRAIN_CRON trong .env).
"""

import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .item_based import train_and_save

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _run_training():
    logger.info("[Scheduler] Starting scheduled training...")
    try:
        result = await train_and_save()
        logger.info("[Scheduler] Training done: %s", result)
    except Exception as e:
        logger.error("[Scheduler] Training failed: %s", e)


def start_scheduler(cron_expr: str = "0 */6 * * *"):
    """
    Khởi động scheduler với cron expression.
    Mặc định: 0 */6 * * * = mỗi 6 tiếng.
    """
    parts = cron_expr.strip().split()
    trigger = CronTrigger(
        minute=parts[0],
        hour=parts[1],
        day=parts[2],
        month=parts[3],
        day_of_week=parts[4],
    )
    scheduler.add_job(_run_training, trigger=trigger, id="retrain_model", replace_existing=True)
    scheduler.start()
    logger.info("[Scheduler] Started. Cron: %s", cron_expr)


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
