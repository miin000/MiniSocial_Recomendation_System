from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class InteractionRecord(BaseModel):
    """Ánh xạ từ collection user_interactions trong MongoDB"""
    user_id: str
    post_id: str
    interaction_type: str   # view | like | comment | share | save
    weight: int
    duration_ms: Optional[int] = None
    created_at: Optional[datetime] = None


class SimilarItem(BaseModel):
    """Một cặp post tương tự"""
    post_id_b: str
    score: float            # Cosine similarity score [0.0 – 1.0]
    based_on: str           # interactions | tags | hybrid


class RecommendedPost(BaseModel):
    """Một bài viết được gợi ý cho user"""
    post_id: str
    score: float
    reason_tag: Optional[str] = None   # VD: "#Technology"
    reason_text: Optional[str] = None  # VD: "Vì bạn thích Technology"
    rank: int


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendedPost]
    generated_at: datetime
    source: str             # "collaborative" | "popular" | "tags" (cold start)


class SimilarResponse(BaseModel):
    post_id: str
    similar_posts: list[SimilarItem]


class TrainStatus(BaseModel):
    status: str             # "ok" | "error"
    interactions_used: int
    posts_covered: int
    users_covered: int
    duration_seconds: float
    trained_at: datetime
    message: str


class EvaluationMetrics(BaseModel):
    hit_rate_at_k: float    # % users nhận được đúng gợi ý trong top-K
    precision_at_k: float   # trung bình hits / K
    ndcg_at_k: float        # normalized discounted cumulative gain


class EvaluateResponse(BaseModel):
    status: str
    k: int
    users_evaluated: int
    total_interactions: int
    metrics: EvaluationMetrics
    interpretation: dict
    reliability: str        # "very_low" | "low" | "medium" | "high"
    duration_seconds: float
    evaluated_at: datetime
