"""
data_loader.py
──────────────
Tải dữ liệu từ MongoDB và chuẩn bị cho việc tính toán ML.
"""
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB  = os.getenv("MONGODB_DB", "minisocial")

_client: Optional[AsyncIOMotorClient] = None


def get_db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URL)
    return _client[MONGODB_DB]


async def load_interactions(since: Optional[datetime] = None) -> pd.DataFrame:
    """
    Tải toàn bộ user_interactions từ MongoDB.
    Nếu `since` được cung cấp, chỉ lấy interactions mới hơn mốc đó.

    Returns DataFrame với columns: user_id, post_id, weight
    (nếu cùng user-post có nhiều loại interaction, lấy sum weight)
    """
    db = get_db()
    query = {}
    if since:
        query["created_at"] = {"$gte": since}

    cursor = db["user_interactions"].find(query, {
        "_id": 0, "user_id": 1, "post_id": 1, "weight": 1
    })
    docs = await cursor.to_list(length=None)

    if not docs:
        return pd.DataFrame(columns=["user_id", "post_id", "weight"])

    df = pd.DataFrame(docs)

    # Gộp weight nếu cùng user-post xuất hiện nhiều lần (nhiều loại interaction)
    df = (
        df.groupby(["user_id", "post_id"], as_index=False)["weight"]
        .sum()
    )
    return df


async def load_post_tags() -> dict[str, list[str]]:
    """
    Tải tags của tất cả bài viết.
    Returns: { post_id: [tag1, tag2, ...] }
    """
    db = get_db()
    cursor = db["posts"].find(
        {"tags": {"$exists": True, "$ne": []}, "status": {"$nin": ["deleted"]}},
        {"_id": 1, "tags": 1}
    )
    docs = await cursor.to_list(length=None)
    return {str(d["_id"]): d.get("tags", []) for d in docs}


async def load_user_tag_preferences(user_id: str) -> list[str]:
    """
    Tính tag nào user hay tương tác nhất (dùng cho cold start / reason text).
    Returns: list[slug] sắp xếp theo tần suất giảm dần
    """
    db = get_db()

    # Lấy post_ids user đã tương tác
    cursor = db["user_interactions"].find(
        {"user_id": user_id},
        {"_id": 0, "post_id": 1, "weight": 1}
    )
    interactions = await cursor.to_list(length=None)
    if not interactions:
        return []

    post_ids = [i["post_id"] for i in interactions]
    weight_map = {i["post_id"]: i["weight"] for i in interactions}

    # Lấy tags của các bài đó
    cursor2 = db["posts"].find(
        {"_id": {"$in": post_ids}},
        {"_id": 1, "tags": 1}
    )
    posts = await cursor2.to_list(length=None)

    tag_score: dict[str, float] = {}
    for p in posts:
        pid = str(p["_id"])
        w = weight_map.get(pid, 1)
        for tag in p.get("tags", []):
            tag_score[tag] = tag_score.get(tag, 0) + w

    return sorted(tag_score, key=lambda t: tag_score[t], reverse=True)


async def load_popular_post_ids(limit: int = 20) -> list[str]:
    """
    Trả về các bài viết phổ biến nhất (fallback khi cold start).
    Xếp theo likes_count + comments_count * 2 + view_count * 0.1
    """
    db = get_db()
    cursor = db["posts"].find(
        {"status": {"$in": ["active", "approved"]}},
        {"_id": 1, "likes_count": 1, "comments_count": 1, "view_count": 1}
    ).sort([("likes_count", -1), ("comments_count", -1)]).limit(limit * 3)

    docs = await cursor.to_list(length=None)

    scored = sorted(
        docs,
        key=lambda d: d.get("likes_count", 0)
                    + d.get("comments_count", 0) * 2
                    + d.get("view_count", 0) * 0.1,
        reverse=True
    )
    return [str(d["_id"]) for d in scored[:limit]]
