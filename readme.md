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


Cốt lõi: "Ai đã tương tác chung"
Model xây User-Item Matrix từ user_interactions:

         post1  post2  post3  post4  post5
acc1       3      2      0      0      0    ← chỉ like sport
acc2       2      3      0      0      0    ← chỉ like sport  
acc3       0      0      4      0      0    ← chỉ like tech
acc4       3      2      0      2      0    ← like sport + news

-> đề xuất news cho acc 1 và acc 2

Cosine similarity giữa post1 và post2:

sim(p1, p2) = 

post1 và post3 có similarity = 0 vì không ai tương tác cả 2.

Vậy model không biết post1 là "sport", chỉ biết acc1 và acc2 đều tương tác post1 và post2 cùng nhau.

Score gợi ý cho user

score(user, candidate_post) = Σ sim(interacted_post, candidate_post) × weight

Ví dụ acc4 đã tương tác post1 (w=3), post2 (w=2):

score(acc4, post4) = sim(post1, post4) × 3 + sim(post2, post4) × 2

→ post4 được gợi ý cho acc4 vì acc4 giống acc1 (cùng like post1, post2), và acc1 cũng like post4.

Kết luận về câu hỏi của bạn
Câu hỏi	Thực tế
"User thích post1 tag sport → gợi ý post sport khác?"	❌ Không trực tiếp — chỉ đúng nếu cùng tập user tương tác các bài sport đó
"Tag có ảnh hưởng score không?"	❌ Không — tag chỉ dùng cho reason_text UI
"Post có cùng tag → similarity cao?"	Chỉ đúng nếu cùng người xem cả 2 bài
Vì sao score = 0.0000 với user của bạn
User 69a4f9620acaecfbd6c3abae nhận Cold Start (popular posts) vì chưa có đủ interactions chồng nhau với user khác để tính similarity. Model không biết bài nào phù hợp → fallback về bài phổ biến nhất, score = 0.

Để có score > 0, cần ít nhất 2 acc cùng like/comment chung ≥ 1 bài — khi đó model mới tính được similarity và đề xuất các bài khác mà user này chưa xem nhưng user kia đã xem.