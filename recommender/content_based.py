"""
content_based.py
────────────────
Content-Based Filtering bổ trợ cho Item-Based Collaborative Filtering.

Thuật toán:
  1. Từ lịch sử tương tác xây dựng Tag Profile của user:
       tag_profile[tag] = sum(weight) của tất cả bài có tag đó mà user đã tương tác
  2. Score mỗi bài chưa tương tác:
       cb_score = sum(tag_profile_norm[tag] for tag in post.tags)
  3. Kết hợp trong item_based.py:
       hybrid_score = CF_score + CONTENT_WEIGHT * cb_score

Lợi ích:
  - Cold-start: user mới chỉ cần 1 interaction để có gợi ý đúng chủ đề
  - Diversity: bổ sung bài cùng tag mà CF chưa phát hiện vì thiếu data
  - Reason accuracy: reason_tag phản ánh đúng tag kết nối user ↔ bài gợi ý
"""

from typing import Optional


def build_tag_profile(
    user_interactions: dict[str, float],   # { post_id: cumulative_weight }
    post_tags: dict[str, list[str]],       # { post_id: [tag1, tag2, ...] }
) -> dict[str, float]:
    """
    Xây dựng tag profile (không normalize) từ tập interaction của 1 user.

    Trả về: { tag_slug: total_weight }
    """
    tag_profile: dict[str, float] = {}
    for pid, weight in user_interactions.items():
        for tag in post_tags.get(pid, []):
            tag_profile[tag] = tag_profile.get(tag, 0.0) + weight
    return tag_profile


def compute_content_scores(
    user_interactions: dict[str, float],   # { post_id: cumulative_weight }
    post_tags: dict[str, list[str]],       # { post_id: [tag1, tag2, ...] }
    interacted_ids: set[str],              # tập bài user đã tương tác (bỏ qua)
) -> dict[str, float]:
    """
    Tính Content-Based score cho tất cả bài chưa tương tác.

    Score của bài P = sum(tag_profile_norm[tag] for tag in P.tags)
    Nghĩa là: bài có nhiều tag trùng với sở thích user → score cao hơn.

    Returns:
        { post_id: cb_score }  — chỉ bao gồm bài có score > 0
    """
    tag_profile = build_tag_profile(user_interactions, post_tags)
    if not tag_profile:
        return {}

    # Normalize theo tổng weight để score nằm trong [0, 1]
    total = sum(tag_profile.values())
    tag_profile_norm = {t: w / total for t, w in tag_profile.items()}

    scores: dict[str, float] = {}
    for pid, tags in post_tags.items():
        if pid in interacted_ids or not tags:
            continue
        score = sum(tag_profile_norm.get(tag, 0.0) for tag in tags)
        if score > 0.0:
            scores[pid] = score

    return scores


def get_best_matching_tag(
    post_tags_list: list[str],
    tag_profile: dict[str, float],
) -> Optional[str]:
    """
    Trả về tag của bài viết (trong post_tags_list) có trọng số cao nhất
    trong tag_profile của user.

    Dùng để tạo reason_text chính xác: "Vì bạn thích #Health"
    thay vì lấy tag đầu tiên tuỳ ý của bài nguồn.

    Trả về None nếu không có tag nào khớp.
    """
    best_tag: Optional[str] = None
    best_score = -1.0
    for tag in post_tags_list:
        s = tag_profile.get(tag, 0.0)
        if s > best_score:
            best_score = s
            best_tag = tag
    return best_tag if best_score > 0.0 else None
