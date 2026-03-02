"""
main.py
───────
FastAPI entry point cho MiniSocial ML Server.

Endpoints:
  GET  /health                        → kiểm tra server còn sống
  POST /train                         → trigger train lại model ngay lập tức
  GET  /recommend/{user_id}           → lấy gợi ý bài viết cho user
  GET  /similar/{post_id}             → lấy bài viết tương tự 1 bài cụ thể
  GET  /interactions/all              → trả về tất cả interactions (debug)

Chạy:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from recommender.item_based import train_and_save, get_recommendations_for_user, get_similar_posts, evaluate_model
from recommender.scheduler import start_scheduler, stop_scheduler
from recommender.data_loader import load_interactions
from models.schemas import TrainStatus, RecommendResponse, SimilarResponse, EvaluateResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RETRAIN_CRON = os.getenv("RETRAIN_CRON", "0 */6 * * *")
PORT         = int(os.getenv("PORT", 8000))


# ── Lifespan: khởi động / tắt scheduler ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ML Server starting up...")
    start_scheduler(RETRAIN_CRON)
    # Train ngay lần đầu khi khởi động server
    try:
        result = await train_and_save()
        logger.info("Initial training: %s", result["message"])
    except Exception as e:
        logger.warning("Initial training skipped: %s", e)
    yield
    logger.info("ML Server shutting down...")
    stop_scheduler()


app = FastAPI(
    title="MiniSocial ML Server",
    description="Item-Based Collaborative Filtering Recommendation System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Chỉnh lại trong production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Kiểm tra server còn sống không"""
    return {"status": "ok", "timestamp": datetime.utcnow()}


@app.post("/train", response_model=TrainStatus)
async def trigger_train():
    """
    Trigger train lại model ngay lập tức.
    Thường được gọi thủ công hoặc sau khi import dữ liệu lớn.
    """
    result = await train_and_save()
    return result


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(user_id: str):
    """
    Trả về danh sách bài viết gợi ý cho user_id.
    
    - Nếu đã có cache chưa hết hạn → trả về ngay (collaborative filtering)
    - Nếu user mới / chưa có dữ liệu → trả về popular posts (cold start)
    """
    result = await get_recommendations_for_user(user_id)
    return result


@app.get("/similar/{post_id}", response_model=SimilarResponse)
async def similar(post_id: str):
    """
    Trả về danh sách bài viết tương tự với post_id.
    Dùng để hiển thị section "Bài viết liên quan" ở trang chi tiết bài.
    """
    result = await get_similar_posts(post_id)
    return result


@app.get("/evaluate")
async def evaluate(k: int = 10):
    """
    Đánh giá độ chính xác model bằng Leave-One-Out Cross-Validation.

    Với mỗi user có >= 2 interactions:
      - Ẩn 1 interaction (test item)
      - Train trên phần còn lại
      - Kiểm tra test item có nằm trong top-K gợi ý không

    Metrics trả về:
      - **hit_rate@K**: bao nhiêu % users nhận gợi ý đúng
      - **precision@K**: trung bình hits / K
      - **ndcg@K**: có tính vị trí xếp hạng (1.0 = hoàn hảo)
    """
    result = await evaluate_model(k=k)
    return result


@app.get("/interactions/stats")
async def interaction_stats():
    """
    Thống kê nhanh về dữ liệu interactions (debug / dashboard).
    """
    df = await load_interactions()
    if df.empty:
        return {"total_interactions": 0, "unique_users": 0, "unique_posts": 0}
    return {
        "total_interactions": len(df),
        "unique_users": df["user_id"].nunique(),
        "unique_posts": df["post_id"].nunique(),
    }


# ── Dev runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
