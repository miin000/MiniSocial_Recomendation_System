"""
item_based.py
─────────────
Nhân tố cốt lõi của hệ thống khuyến nghị Item-Based Collaborative Filtering.

Thuật toán:
  1. Đọc user_interactions → xây dựng User-Item Matrix (user × post, giá trị = tổng weight)
  2. Tính Cosine Similarity giữa tất cả cặp post (item-item)
  3. Với mỗi user:
       - Lấy danh sách post đã tương tác
       - Tìm top-N post tương tự nhất (chưa tương tác)
       - Xếp hạng theo điểm tổng = sum(similarity × weight)
  4. Ghi kết quả vào item_similarity và user_recommendations trong MongoDB
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

from .data_loader import get_db, load_interactions, load_post_tags, load_popular_post_ids, load_user_tag_preferences

load_dotenv()

TOP_N_SIMILAR        = int(os.getenv("TOP_N_SIMILAR", 10))
TOP_N_RECOMMENDATIONS = int(os.getenv("TOP_N_RECOMMENDATIONS", 20))
MIN_INTERACTIONS     = int(os.getenv("MIN_INTERACTIONS", 2))
RECOMMEND_EXPIRES_HOURS = 6

logger = logging.getLogger(__name__)


def _build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot DataFrame thành ma trận user × post.
    Giá trị = tổng weight interactions. NaN → 0.
    """
    matrix = df.pivot_table(
        index="user_id",
        columns="post_id",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )
    return matrix


def _compute_item_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Tính Cosine Similarity giữa các post (cột của matrix).
    Returns DataFrame (post × post) với score [0, 1].
    """
    # Transpose: mỗi hàng là 1 post, mỗi cột là 1 user
    item_matrix = matrix.T.values  # shape: (n_posts, n_users)
    sim_matrix = cosine_similarity(item_matrix)
    return pd.DataFrame(
        sim_matrix,
        index=matrix.columns,
        columns=matrix.columns,
    )


async def train_and_save() -> dict:
    """
    Hàm chính: train model và lưu kết quả vào MongoDB.
    Được gọi bởi scheduler hoặc POST /train endpoint.
    """
    start = time.time()
    db = get_db()

    # ── 1. Load dữ liệu ──────────────────────────────────────────────────────
    df = await load_interactions()

    if df.empty or len(df) < MIN_INTERACTIONS:
        logger.warning("Not enough interactions to train (%d rows)", len(df))
        return {
            "status": "skipped",
            "interactions_used": len(df),
            "posts_covered": 0,
            "users_covered": 0,
            "duration_seconds": round(time.time() - start, 2),
            "trained_at": datetime.utcnow(),
            "message": f"Need at least {MIN_INTERACTIONS} interactions, got {len(df)}",
        }

    # ── 2. Xây matrix & tính similarity ──────────────────────────────────────
    matrix = _build_user_item_matrix(df)
    sim_df  = _compute_item_similarity(matrix)

    post_ids = list(sim_df.index)
    users    = list(matrix.index)

    # ── 3. Lưu item_similarity ───────────────────────────────────────────────
    now = datetime.utcnow()
    sim_docs = []
    for i, pid_a in enumerate(post_ids):
        # Lấy top-N similar (bỏ qua chính nó)
        scores = sim_df.iloc[i].drop(index=pid_a).nlargest(TOP_N_SIMILAR)
        for pid_b, score in scores.items():
            if score > 0:
                sim_docs.append({
                    "post_id_a": pid_a,
                    "post_id_b": pid_b,
                    "score": round(float(score), 6),
                    "based_on": "interactions",
                    "computed_at": now,
                })

    if sim_docs:
        # Xóa cũ, ghi mới
        await db["item_similarity"].delete_many({})
        await db["item_similarity"].insert_many(sim_docs)
        logger.info("Saved %d similarity pairs", len(sim_docs))

    # ── 4. Tạo user_recommendations ───────────────────────────────────────────
    expires_at = now + timedelta(hours=RECOMMEND_EXPIRES_HOURS)
    post_tags  = await load_post_tags()
    rec_docs   = []

    for user_id in users:
        user_row        = matrix.loc[user_id]
        interacted_ids  = set(user_row[user_row > 0].index)

        # Điểm tổng cho mỗi post chưa tương tác
        candidate_scores: dict[str, float] = {}
        for interacted_pid in interacted_ids:
            if interacted_pid not in sim_df.index:
                continue
            w = float(user_row[interacted_pid])
            sim_row = sim_df.loc[interacted_pid]
            for candidate_pid, sim_score in sim_row.items():
                if candidate_pid in interacted_ids:
                    continue
                candidate_scores[candidate_pid] = (
                    candidate_scores.get(candidate_pid, 0.0)
                    + sim_score * w
                )

        if not candidate_scores:
            continue

        # Top-N
        top_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )[:TOP_N_RECOMMENDATIONS]

        # Tính lý do gợi ý (tag nổi bật nhất)
        user_top_tags = await load_user_tag_preferences(user_id)
        reason_tag  = f"#{user_top_tags[0].capitalize()}" if user_top_tags else None
        reason_text = f"Vì bạn thích {user_top_tags[0].capitalize()}" if user_top_tags else "Dành cho bạn"

        for rank, (pid, score) in enumerate(top_candidates, start=1):
            rec_docs.append({
                "user_id":     user_id,
                "post_id":     pid,
                "score":       round(float(score), 6),
                "reason_tag":  reason_tag,
                "reason_text": reason_text,
                "rank":        rank,
                "generated_at": now,
                "expires_at":   expires_at,
            })

    if rec_docs:
        await db["user_recommendations"].delete_many({})
        await db["user_recommendations"].insert_many(rec_docs)
        logger.info("Saved recommendations for %d users", len(users))

    duration = round(time.time() - start, 2)
    return {
        "status": "ok",
        "interactions_used": len(df),
        "posts_covered": len(post_ids),
        "users_covered": len(users),
        "duration_seconds": duration,
        "trained_at": now,
        "message": f"Trained successfully in {duration}s",
    }


async def get_recommendations_for_user(user_id: str) -> dict:
    """
    Lấy gợi ý từ cache user_recommendations.
    Nếu không có hoặc đã hết hạn → trả về popular posts (cold start).
    """
    db = get_db()
    now = datetime.utcnow()

    recs = await db["user_recommendations"].find(
        {"user_id": user_id, "expires_at": {"$gt": now}},
        {"_id": 0, "post_id": 1, "score": 1, "reason_tag": 1, "reason_text": 1, "rank": 1}
    ).sort("rank", 1).to_list(length=TOP_N_RECOMMENDATIONS)

    if recs:
        return {
            "user_id": user_id,
            "recommendations": recs,
            "generated_at": now,
            "source": "collaborative",
        }

    # Cold start: trả về popular posts
    popular = await load_popular_post_ids(TOP_N_RECOMMENDATIONS)
    return {
        "user_id": user_id,
        "recommendations": [
            {"post_id": pid, "score": 0.0, "reason_tag": None,
             "reason_text": "Bài viết phổ biến", "rank": i + 1}
            for i, pid in enumerate(popular)
        ],
        "generated_at": now,
        "source": "popular",
    }


async def evaluate_model(k: int = 10) -> dict:
    """
    Đánh giá độ chính xác model bằng Leave-One-Out Cross-Validation (LOO-CV).

    Thuật toán:
      - Với mỗi user có >= 2 interactions:
          1. Ẩn interaction có weight cao nhất ("test item")
          2. Train item-similarity trên phần còn lại
          3. Tạo top-K gợi ý
          4. Kiểm tra test item có trong top-K không
      - Tính Hit Rate@K, Precision@K, NDCG@K

    Lưu ý: cần ít nhất 5 users có >= 2 interactions để kết quả có ý nghĩa thống kê.
    """
    start = time.time()
    df = await load_interactions()

    if df.empty:
        return {
            "status": "error",
            "message": "Không có dữ liệu interactions",
            "users_evaluated": 0,
        }

    # Load raw interactions (user_id, post_id, weight) trước khi group
    db = get_db()
    raw_cursor = db["user_interactions"].find({}, {"_id": 0, "user_id": 1, "post_id": 1, "weight": 1})
    raw_docs = await raw_cursor.to_list(length=None)
    raw_df = pd.DataFrame(raw_docs) if raw_docs else pd.DataFrame(columns=["user_id", "post_id", "weight"])

    # Nhóm theo user_id + post_id (sum weight)
    raw_df = raw_df.groupby(["user_id", "post_id"], as_index=False)["weight"].sum()

    # Chỉ đánh giá user có >= 2 interactions
    user_counts = raw_df.groupby("user_id")["post_id"].count()
    eligible_users = user_counts[user_counts >= 2].index.tolist()

    if not eligible_users:
        return {
            "status": "insufficient_data",
            "message": "Cần ít nhất 1 user có >= 2 interactions để đánh giá",
            "users_evaluated": 0,
            "k": k,
            "reliability": "low",
        }

    hits = 0
    ndcg_scores = []
    precision_scores = []

    for user_id in eligible_users:
        user_data = raw_df[raw_df["user_id"] == user_id].copy()

        # Ẩn interaction có weight cao nhất làm test item
        test_row = user_data.nlargest(1, "weight").iloc[0]
        test_post_id = test_row["post_id"]
        train_data = user_data[user_data["post_id"] != test_post_id]

        if train_data.empty:
            continue

        # Combine với interactions của các user khác
        other_users = raw_df[raw_df["user_id"] != user_id]
        train_df = pd.concat([train_data, other_users], ignore_index=True)

        if len(train_df) < 2:
            continue

        # Build matrix & similarity trên training data
        try:
            matrix = _build_user_item_matrix(train_df)
        except Exception:
            continue

        if user_id not in matrix.index:
            continue

        sim_df = _compute_item_similarity(matrix)
        user_row = matrix.loc[user_id]
        interacted_ids = set(user_row[user_row > 0].index)

        # Tính điểm cho candidates
        candidate_scores: dict[str, float] = {}
        for pid in interacted_ids:
            if pid not in sim_df.index:
                continue
            w = float(user_row[pid])
            for candidate_pid, sim_score in sim_df.loc[pid].items():
                if candidate_pid in interacted_ids:
                    continue
                candidate_scores[candidate_pid] = (
                    candidate_scores.get(candidate_pid, 0.0) + sim_score * w
                )

        # Xếp hạng top-K
        top_k = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_ids = [pid for pid, _ in top_k]

        # Tính metrics
        hit = 1 if test_post_id in top_k_ids else 0
        hits += hit
        precision_scores.append(hit / k)

        # NDCG: nếu hit, điểm cao hơn nếu xếp hạng cao hơn
        if hit:
            rank = top_k_ids.index(test_post_id) + 1  # 1-based
            ndcg_scores.append(1.0 / np.log2(rank + 1))
        else:
            ndcg_scores.append(0.0)

    n = len(eligible_users)
    evaluated = len(precision_scores)

    if evaluated == 0:
        return {
            "status": "insufficient_data",
            "message": "Không tính được metrics do dữ liệu quá ít",
            "users_evaluated": 0,
        }

    hit_rate   = round(hits / evaluated, 4)
    precision  = round(float(np.mean(precision_scores)), 4)
    ndcg       = round(float(np.mean(ndcg_scores)), 4)
    duration   = round(time.time() - start, 2)

    # Đánh giá độ tin cậy
    if evaluated < 5:
        reliability = "very_low — cần thêm dữ liệu (< 5 users đủ điều kiện)"
    elif evaluated < 20:
        reliability = "low — kết quả sơ bộ (< 20 users)"
    elif evaluated < 100:
        reliability = "medium"
    else:
        reliability = "high"

    return {
        "status": "ok",
        "k": k,
        "users_evaluated": evaluated,
        "total_interactions": len(raw_df),
        "metrics": {
            "hit_rate_at_k":    hit_rate,   # % users có test item trong top-K
            "precision_at_k":  precision,  # trung bình hits/K
            "ndcg_at_k":       ndcg,       # normalized discounted cumulative gain
        },
        "interpretation": {
            "hit_rate": f"{hit_rate * 100:.1f}% users nhận được gợi ý đúng trong top-{k}",
            "ndcg":     "1.0 = hoàn hảo, 0.0 = tệ — xét vị trí xếp hạng",
            "verdict":  "Tốt" if hit_rate >= 0.4 else ("Trung bình" if hit_rate >= 0.2 else "Cần thêm dữ liệu"),
        },
        "reliability": reliability,
        "duration_seconds": duration,
        "evaluated_at": datetime.utcnow(),
    }


async def get_similar_posts(post_id: str) -> dict:
    """
    Lấy danh sách bài tương tự từ cache item_similarity.
    """
    db = get_db()

    docs = await db["item_similarity"].find(
        {"post_id_a": post_id},
        {"_id": 0, "post_id_b": 1, "score": 1, "based_on": 1}
    ).sort("score", -1).limit(TOP_N_SIMILAR).to_list(length=None)

    return {
        "post_id": post_id,
        "similar_posts": docs,
    }
