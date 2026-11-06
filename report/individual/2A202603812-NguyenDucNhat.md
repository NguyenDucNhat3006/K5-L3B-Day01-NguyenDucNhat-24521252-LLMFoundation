# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603812
- **Họ và tên:** Nguyễn Đức Nhật
- **SEED** (in ở ô đầu notebook): 7406

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.png`](hand_calc.png) — ảnh tính tay có các bước | [x] |
## 2. Kết quả chính

| Phương pháp | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.810 | 0.895 | 0.974 | 6.552 | 27,892.037 | 0.052 | 0.189 | 0.0 |
| **2. BoW + MLP (deep learning)** | 0.989 | 0.910 | 0.948 | 1.000 | 10.598 | 26,406.315 | 0.306 | 0.359 | 0.0 |
| **3a. Encoder (MiniLM) + kNN** | 0.812 | 0.950 | 0.876 | 0.976 | 119.348 | 22.731 | 51.811 | 61.266 | 0.0 |
| **3b. Encoder (MiniLM) + LogReg** | 0.931 | 0.950 | 0.941 | 0.976 | 119.794 | 32.726 | 30.852 | 41.373 | 0.0 |
| **3c. Encoder (MiniLM) + MLP head** | 0.922 | 0.950 | 0.936 | 0.976 | 120.384 | 33.740 | 48.067 | 61.105 | 0.0 |

## 3. Dự đoán và thực tế

1. **ml_f1 (F1 trên tập test của BoW):**
   - *Dự đoán:* F1 ~0.7, precision cao hơn recall vì dữ liệu mẫu lệch về số lượng mail ham, khiến mô hình có xu hướng dự đoán mail ham cao hơn, làm độ phủ (recall) giảm.
   - *Thực tế:* F1 thực tế đạt 0.895 (precision 1.000, recall 0.810). Precision cao tuyệt đối vì từ điển Bag-of-Words bắt rất chuẩn các từ đặc trưng của spam, nhưng recall thấp hơn do các tin spam né từ khóa hoặc cấu trúc câu lạ không có trong tập train.

2. **ml_lr (Đường loss khi lr = 0.001):**
   - *Dự đoán:* Loss có hình dạng dao động mạnh giống nhịp tim trên máy đo điện tim.
   - *Thực tế:* Với lr quá nhỏ, bước cập nhật quá bé khiến đường loss giảm cực kỳ chậm và phẳng lỳ thay vì dao động mạnh (dao động mạnh xảy ra khi lr quá lớn như lr = 50).

3. **emb_search (Tìm kiếm tương đồng nghĩa):**
   - *Dự đoán:* Kết quả truy vấn gần nhất là spam.
   - *Thực tế:* Đúng hoàn toàn. Dù câu truy vấn không chứa từ khóa nhạy cảm như "free" hay "win", mô hình embedding MiniLM vẫn bắt được ý nghĩa tương đồng về mặt ngữ nghĩa (chương trình khuyến mãi/quà tặng) và kéo các tin spam lên top đầu.

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** Mô hình Bag-of-Words (Phần 1) tụt nặng nhất ở nhóm **Né từ khoá (obfuscation)** và **Tưởng spam nhưng không phải (ham_looks_spam)**.
- **2 case cụ thể và cơ chế sai:**
  1. *Case 1 (Né từ khoá):* `Fr€€ ringt0nes 4 ur m0bile!!!` được viết thay ký tự ($€, 0, 4$). Mô hình BoW không tìm thấy các token chuẩn (`free`, `for`) trong từ điển nên bỏ sót hoàn toàn và gán nhãn là ham.
  2. *Case 2 (Tưởng spam nhưng không phải):* `URGENT: mum's car broke down near the station, can you pick her up? Call me asap`. Mô hình dính bẫy từ khóa `URGENT` và `Call` (v vốn có trọng số đẩy về spam rất cao trong tập train), dẫn đến việc phân loại nhầm tin nhắn khẩn cấp của người nhà thành tin rác.

## 5. Thử thách cá nhân

- **Cách làm:** Xử lý yêu cầu sửa code được phân công theo SEED (7406) trong thời gian quy định, rà soát lại các hàm vector hóa, hàm tính toán hàm mất mát và cập nhật gradient để đảm bảo không sử dụng thư viện ngoài `sklearn` trong các vùng cấm tự code.
- **Kết quả & Giải thích:** Các kiểm thử tự động (`_run_check`) đều vượt qua thành công, các ma trận và số chiều tensor trả về đúng yêu cầu kỹ thuật, đảm bảo tính đúng đắn của thuật toán gradient descent thủ công.

## 6. Phương án cho sản phẩm thật

- **Lựa chọn:** Chọn phương án **3b (Encoder MiniLM + LogReg)** hoặc **2 (BoW + MLP)** tùy theo ràng buộc phần cứng, nhưng ưu tiên **3b** nếu xét về độ bền vững trước các biến thể từ ngữ.
- **Dẫn số liệu:** Phương án 3b đạt p95 latency khoảng **41.37 ms**, nằm hoàn toàn trong ngưỡng SLA yêu cầu ($< 50\text{ ms/tin}$), đồng thời xử lý tốt các trường hợp lách từ khóa nhờ không gian vector ngữ nghĩa.
- **Đánh đổi:** Chấp nhận đánh đổi thời gian khởi động ban đầu (mất khoảng ~119 giây để mã hóa/embedding tập dữ liệu lớn) và yêu cầu tài nguyên tính toán cao hơn so với mô hình BoW truyền thống thuần túy.

## 7. AI log và điều đã học

- **AI Log:**
  - *Lần 1:* Hỏi AI cách tối ưu phép nhân ma trận trong hàm `top_k_similar` để tránh dùng vòng lặp Python qua $n$ dòng. Nhận được hướng dẫn chuẩn hóa vector hàng và dùng phép nhân vô hướng ma trận `@`. Đã áp dụng và kiểm tra qua unit test.
  - *Lần 2:* Nhờ AI giải thích cách cấu hình tham số `provider_kwargs` cho các model reasoning mới của OpenAI để tránh lỗi 400. Nhận được cấu trúc truyền `max_completion_tokens` kết hợp `reasoning_effort="none"`. Đã kiểm thử thành công.
- **Điều rút ra:** Hôm nay tôi hiểu sâu hơn về sự tiến hóa của NLP qua 4 thế hệ: từ việc đếm tần suất từ thô sơ (Machine Learning) phụ thuộc hoàn toàn vào từ khóa, đến việc biểu diễn ý nghĩa câu trừu tượng trong không gian nhiều chiều (Embedding), giúp hệ thống nhận diện được ngữ cảnh ngay cả khi câu chữ đã bị biến đổi tinh vi.