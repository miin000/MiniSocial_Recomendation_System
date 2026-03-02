cd ml_server
pip install -r requirements.txt
python main.py

Giải thích các chỉ số Recommendation System
📊 Interaction Stats (Dữ liệu tương tác)
Chỉ số	Ý nghĩa
Total Interactions	Tổng số lần user tương tác (like, view, comment, share) với bài viết
Unique Users	Số user khác nhau đã có ít nhất 1 tương tác
Unique Posts	Số bài viết khác nhau đã được tương tác
Càng nhiều → model train càng chính xác. Cần tối thiểu 50+ interactions để có kết quả đáng tin.

🤖 Model Training
Chỉ số	Ý nghĩa
Interactions Used	Số interactions thực sự dùng để train (sau khi lọc noise)
Posts in Model	Số bài viết model đã học được pattern
Users in Model	Số user model có thể đưa ra gợi ý
Duration	Thời gian train (giây) — thường < 1s với data nhỏ
🎯 Đánh giá độ chính xác (Evaluate @K)
K = số bài gợi ý tối đa (ví dụ K=10 → gợi ý 10 bài)

Hit Rate@K
Giá trị	Đánh giá
≥ 40%	🟢 Tốt
20–40%	🟡 Trung bình
< 20%	🔴 Cần thêm dữ liệu
Ví dụ: Hit Rate@10 = 0.6 → 60% user có ít nhất 1 bài đúng trong 10 bài gợi ý

Precision@K
Giá trị	Đánh giá
≥ 0.05	🟢 Tốt
0.02–0.05	🟡 Trung bình
< 0.02	🔴 Kém
Ví dụ: Precision@10 = 0.3 → trung bình 3/10 bài gợi ý là đúng (30%)

NDCG@K (Normalized Discounted Cumulative Gain)
Giá trị	Đánh giá
≥ 0.3	🟢 Tốt
0.1–0.3	🟡 Trung bình
< 0.1	🔴 Kém
Ví dụ: Gợi ý đúng ở vị trí 1 = điểm cao hơn gợi ý đúng ở vị trí 10

🔒 Độ tin cậy (Reliability)
Mức	Điều kiện	Ý nghĩa
very_low	< 5 users đủ điều kiện	Chưa đáng tin, bỏ qua kết quả
low	5–20 users	Tham khảo sơ bộ
medium	20–100 users	Tương đối tin cậy
high	≥ 100 users	✅ Đáng tin cậy

Hệ thống gợi ý 10 bài cho user A:
├── Hit Rate: A có thấy bài mình thích không? (có/không)
├── Precision: Trong 10 bài đó, mấy bài A thực sự thích?
└── NDCG: Bài A thích có được xếp lên đầu không?