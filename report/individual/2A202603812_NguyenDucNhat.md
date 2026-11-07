# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603812
- **Họ và tên:** Nguyễn Đức Nhật
- **SEED** : 7406

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |

## 2. Kết quả chính

| Phương pháp | Precision | Recall | F1 test | F1 SUB 40 | Train s | Tin/s cả lô | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. BoW + LogReg | 1.000 | 0.810 | 0.895 | 0.974 | 12.423 | 6964.438 | 0.055 | 0.220 | 0.0 |
| 2. BoW + MLP | 0.989 | 0.910 | 0.948 | 1.000 | 13.769 | 17507.169 | 0.260 | 0.421 | 0.0 |
| 3a. Encoder MiniLM + kNN | 0.812 | 0.950 | 0.876 | 0.976 | 124.101 | 20.280 | 51.988 | 61.028 | 0.0 |
| 3b. Encoder MiniLM + LogReg | 0.931 | 0.950 | 0.941 | 0.976 | 124.502 | 29.449 | 30.465 | 38.412 | 0.0 |
| 3c. Encoder MiniLM + MLP head | 0.922 | 0.950 | 0.936 | 0.976 | 125.087 | 29.920 | 31.297 | 44.714 | 0.0 |
| 4a. LLM nhỏ zero-shot | 0.800 | 1.000 | - | 0.889 | 0.000 | 1.123 | 320.255 | 2438.817 | 0.0 |
| 4b. LLM lớn zero-shot | 0.833 | 1.000 | - | 0.909 | 0.000 | 1.266 | 323.722 | 2329.826 | 0.0 |
| 4c. LLM nhỏ few-shot | 0.870 | 1.000 | - | 0.930 | 0.000 | 0.671 | 2208.554 | 2414.748 | 0.0 |

## 3. Dự đoán và thực tế

1. Dự đoán ml_f1 lúc 12:56:15 và thực tế:
   - Dự đoán: F1 khoảng 0.7, precision cao hơn recall vì dữ liệu mẫu bị lệch nhiều về tin ham nên mô hình có xu hướng đoán ham nhiều hơn làm độ phủ giảm.
   - Thực tế: F1 đạt 0.895 với precision bằng 1.000 và recall bằng 0.810. Chiều hướng precision lớn hơn recall đúng như dự đoán nhưng F1 thực tế cao hơn kỳ vọng vì các từ khóa đặc trưng của tin rác xuất hiện rất rõ trong từ điển 2000 từ. Mô hình chỉ bỏ sót khoảng 19 phần trăm tin rác có từ vựng lạ hoặc viết biến thể.

2. Dự đoán ml_lr lúc 12:56:29 và thực tế:
   - Dự đoán: Khi tốc độ học bằng 0.001 thì đường loss dao động mạnh giống nhịp tim trên máy đo điện tim.
   - Thực tế: Kết quả ngược lại với dự đoán ban đầu. Khi tốc độ học bằng 0.001 quá nhỏ, bước cập nhật trọng số mỗi vòng rất bé nên đường loss giảm chậm và gần như phẳng. Hiện tượng đường loss nhảy vọt qua lại và dao động mạnh chỉ xảy ra khi tốc độ học quá lớn ở mức 50.

3. Dự đoán emb_tricky lúc 13:01:49 và thực tế:
   - Dự đoán: Mô hình BoW sẽ dự đoán đúng nhiều hơn trên các câu ngụy trang.
   - Thực tế: Dự đoán sai vì BoW chỉ đúng 5 trên 6 câu còn Encoder cộng LogReg đúng tuyệt đối 6 trên 6 câu. Với những câu thay thế từ khóa bằng cách diễn đạt đồng nghĩa hoặc nhắc khéo chuyện tài khoản hết hạn, BoW chỉ đếm mặt chữ rời rạc nên bỏ lọt, trong khi mô hình MiniLM nắm được ngữ cảnh cả câu nên phân loại chính xác.

4. Dự đoán llm_models lúc 13:01:50 và llm_fewshot lúc 13:03:01 so với thực tế:
   - Dự đoán: Mô hình lớn có F1 cao hơn nhẹ so với mô hình nhỏ, thêm 4 ví dụ mẫu giúp F1 tăng nhẹ nhưng tốn chi phí và thời gian hơn, đồng thời F1 của LLM ngang ngửa mô hình 3b trên tập SUB.
   - Thực tế: Mô hình LLM lớn đạt F1 là 0.909 cao hơn mức 0.889 của mô hình nhỏ, và khi thêm 4 ví dụ mẫu ở bài 4c thì F1 tăng lên 0.930 đúng như dự đoán. Tuy nhiên trên tập SUB 40 tin chuẩn thì LLM vẫn thấp hơn mức 0.976 của nhóm MiniLM và có độ trễ p95 lên tới hơn 2300 ms mỗi tin.

## 4. Phân tích 50 case khó

- Mô hình tụt điểm nhiều nhất: Mô hình 1 BoW cộng LogReg tụt mạnh nhất khi F1 giảm từ 0.974 trên tập SUB xuống chỉ còn 0.286 trên 50 case khó và độ chính xác chỉ đạt 50 phần trăm. Mô hình 2 BoW cộng MLP cũng tụt từ 1.000 xuống 0.421. Nhóm làm khó BoW nhất là spam giọng tự nhiên với tỷ lệ đoán đúng 0 phần trăm, thao túng lừa đảo đúng 12 phần trăm và né từ khóa đúng 17 phần trăm.
- Hai case cụ thể và cơ chế gây sai:
  1. Case số 32 thuộc nhóm né từ khóa có nội dung Fr€€ ringt0nes 4 ur m0bile!!! txt T0NE to 85555 now với nhãn thật là spam nhưng BoW đoán nhầm thành ham. Người gửi đã thay chữ e bằng ký hiệu tiền tệ và thay chữ o bằng số 0. Bộ tách từ chỉ cắt theo mặt chữ nên tạo ra các từ lạ không khớp với các từ free, ringtones hay mobile trong từ điển 2000 từ của tập huấn luyện, làm vector đầu vào gần như toàn số 0 và kéo xác suất dự đoán xuống dưới ngưỡng 0.5.
  2. Case số 21 thuộc nhóm thao túng lừa đảo có nội dung giả làm con nhắn cho mẹ báo rơi điện thoại và nhờ chuyển gấp 200 bảng vào tài khoản bạn với nhãn thật là spam nhưng BoW và cả kNN dễ đoán nhầm. Câu tin nhắn dùng toàn bộ từ vựng giao tiếp gia đình hàng ngày và không hề chứa các từ khóa quảng cáo trúng thưởng quen thuộc, khiến mô hình đếm từ cộng dồn các trọng số âm của nhóm từ đời thường và kết luận nhầm là tin nhắn bình thường.

## 5. Thử thách cá nhân

- Đề bài thực hiện: Đề số 3 yêu cầu thay đổi lớp ẩn của mạng MLP thành 4 và 512 nơ-ron so với mức 64 mặc định, báo cáo số lượng tham số, điểm F1 trên tập kiểm tra, thời gian huấn luyện và giải thích khi nào nên chọn mạng nhỏ.
- Cách làm và kết quả chạy thực nghiệm: Thực hiện vòng lặp khởi tạo lại seed và huấn luyện mạng MLP qua 30 vòng lặp cho ba cấu hình kích thước lớp ẩn là 4, 64 và 512 nơ-ron trên cùng tập dữ liệu Bag of Words 2000 chiều. Kết quả thu được như sau:

| Kích thước lớp ẩn | Số lượng tham số | Thời gian train s | Precision | Recall | F1 test |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 4 nơ-ron | 8009 | 2.68 | 1.0000 | 0.9100 | 0.9529 |
| 64 nơ-ron | 128129 | 5.21 | 0.9891 | 0.9100 | 0.9479 |
| 512 nơ-ron | 1025025 | 28.52 | 0.9787 | 0.9200 | 0.9485 |

- Giải thích kết quả và khi nào nên chọn mạng nhỏ:
  - Mạng nhỏ 4 nơ-ron đạt điểm F1 cao nhất là 0.9529 và giữ được precision tuyệt đối 1.0000 dù chỉ có 8009 tham số và mất 2.68 giây để học. Nguyên nhân là dữ liệu đầu vào Bag of Words trên bài toán lọc tin rác có ranh giới phân chia khá đơn giản, lớp ẩn nhỏ 4 nơ-ron giúp ép mô hình chỉ giữ lại những tín hiệu quan trọng nhất và hạn chế học thuộc nhiễu. Ngược lại mạng 512 nơ-ron có hơn 1 triệu tham số trên tập huấn luyện chỉ có 3619 mẫu nên bị dư thừa dung lượng, làm giảm precision xuống 0.9787 và tốn thời gian huấn luyện gấp hơn 10 lần so với mạng 4 nơ-ron.
  - Nên chọn mạng nhỏ khi tập dữ liệu huấn luyện có quy mô nhỏ hoặc vừa, đặc trưng đầu vào đơn giản dễ dẫn đến học vẹt nếu dùng mạng lớn, hoặc khi cần triển khai thực tế trên phần cứng hạn chế bộ nhớ và yêu cầu tốc độ huấn luyện cũng như phản hồi cực nhanh.

## 6. Phương án cho sản phẩm thật

- Lựa chọn cho hệ thống lọc 10 triệu tin nhắn mỗi ngày với yêu cầu độ trễ p95 dưới 50 ms mỗi tin: Kết hợp mô hình 2 BoW cộng MLP và mô hình 3b Encoder MiniLM cộng LogReg theo kiến trúc lọc hai tầng, hoặc chọn 3b làm bộ lọc chính nếu hạ tầng có GPU xử lý theo lô.
- Dẫn số liệu từ bảng thực nghiệm:
  - Xét các mô hình thỏa mãn độ trễ p95 dưới 50 ms gồm có mô hình 1 đạt 0.220 ms, mô hình 2 đạt 0.421 ms, mô hình 3b đạt 38.412 ms và mô hình 3c đạt 44.714 ms. Trong khi đó mô hình 3a kNN mất 61.028 ms và các mô hình LLM mất từ 2329 ms đến 2438 ms đều vượt quá giới hạn cho phép.
  - Mô hình 2 BoW cộng MLP có tốc độ xử lý cả lô rất cao đạt 17507 tin mỗi giây và F1 trên tập test thường đạt 0.948, nhưng khi gặp tập 50 case khó thì F1 rơi xuống 0.421. Ngược lại mô hình 3b MiniLM cộng LogReg vừa đạt chuẩn độ trễ với p95 bằng 38.412 ms vừa giữ được recall 0.950 trên tập test thường và đạt F1 0.692 trên tập 50 case khó.
  - Về chi phí vận hành, cả mô hình 2 và 3b đều chạy nội bộ không tốn phí gọi API theo từng tin nhắn, trong khi nếu dùng LLM nhỏ cho 10 triệu tin nhắn mỗi ngày theo mức tính tay sẽ tốn khoảng 105 đô la mỗi ngày tương đương 3150 đô la mỗi tháng.
- Đánh đổi khi triển khai: Mô hình 3b tốn khoảng 124.5 giây để mã hóa tập huấn luyện ban đầu và thông lượng xử lý tuần tự trên CPU đạt khoảng 29.45 tin mỗi giây, thấp hơn mức trung bình 116 tin mỗi giây của 10 triệu tin mỗi ngày nếu chỉ chạy đơn luồng. Do đó giải pháp thực tế hiệu quả nhất là đặt mô hình 2 BoW cộng MLP ở tầng đầu để lọc nhanh 90 phần trăm các tin nhắn rõ ràng trong 0.42 ms, còn khoảng 10 phần trăm các tin nhắn có xác suất nằm ở vùng nghi ngờ từ 0.2 đến 0.8 mới chuyển sang tầng hai dùng mô hình 3b MiniLM cộng LogReg hoặc kiểm tra sâu bằng LLM có prompt phòng thủ.

## 7. AI log và điều đã học

- Nhật ký sử dụng AI:
  - Lần 1 ở phần 3: Hỏi AI cách viết hàm tìm k vector tương đồng nhất bằng phép nhân ma trận trong NumPy để không phải dùng vòng lặp duyệt qua từng dòng của ma trận E_train. AI hướng dẫn chuẩn hóa độ dài vector truy vấn và từng hàng của ma trận về 1 rồi dùng toán tử nhân ma trận kết hợp hàm argsort lật ngược chiều. Đã tự kiểm tra trường hợp vector 0 và chạy qua bài kiểm tra tự động 3.1 và 3.2.
  - Lần 2 ở phần 4: Hỏi AI cách truyền tham số giới hạn token và cấu hình provider cho các mô hình suy luận trên Groq và OpenAI trong hàm call_openai, cũng như cách làm sạch chuỗi kết quả trả về trong hàm parse_label. AI gợi ý truyền từ khóa mở rộng provider_kwargs và dùng biểu thức chính quy cắt bỏ phần tiền tố nhãn trước khi so sánh. Đã chạy thử nghiệm thực tế với API của Groq và vượt qua các bài kiểm tra tự động từ 4.1 đến 4.4.
- Điều rút ra sau buổi thực hành: Qua việc thử nghiệm cả 4 thế hệ mô hình trên cùng một bài toán, em nhận ra điểm F1 cao trên tập kiểm tra thông thường chưa phản ánh hết khả năng hoạt động ngoài thực tế vì dữ liệu cùng phân phối rất dễ học thuộc từ khóa. Những mô hình đơn giản như Logistic Regression khi kết hợp với bộ trích xuất đặc trưng ngữ nghĩa tốt như MiniLM có thể đạt hiệu quả rất cao và đáp ứng được yêu cầu độ trễ khắt khe, trong khi LLM hiểu ngữ cảnh và chống lách từ khóa tốt nhất nhưng cần có prompt phòng thủ để tránh bị thao túng câu lệnh và chỉ nên dùng chọn lọc do độ trễ lớn.
