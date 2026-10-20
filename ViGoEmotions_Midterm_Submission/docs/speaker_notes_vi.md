# Speaker notes (tiếng Việt) — ViGoEmotions Midterm

Cùng nội dung với phần Notes trong file PPTX và deck web.

## Slide 1 — ViGoEmotions

Chào thầy/cô và các bạn. Nhóm em làm Track 2: ViGoEmotions — nhận diện cảm xúc đa nhãn chi tiết cho bình luận mạng xã hội tiếng Việt. Bài midterm gồm: hiểu dữ liệu, giao thức chia tập và tiền xử lý, ba mô hình bắt buộc (TF-IDF + Logistic Regression, BiLSTM, PhoBERT) cộng thêm các encoder tiếng Việt mạnh hơn, trả lời 5 câu hỏi midterm, phân tích lỗi và đề xuất câu hỏi nghiên cứu cho Final.

## Slide 2 — One comment can carry several emotions at once

Bài toán là phân loại đa nhãn: một bình luận có thể mang nhiều cảm xúc cùng lúc, ví dụ minh họa này (nhóm tự viết, không lấy từ dataset) vừa buồn, vừa thương, vừa quan tâm. Vì vậy mô hình đưa ra 28 quyết định nhị phân độc lập — 27 cảm xúc của GoEmotions cộng nhãn neutral — với sigmoid cho từng nhãn chứ không phải softmax. Theo đề, metric chính là Macro-F1 và Micro-F1; subset accuracy quá khắt khe nên chỉ báo cáo phụ. Khó khăn đặc thù là tiếng Việt phi chuẩn: teencode, tiếng lóng, emoji.

## Slide 3 — 20,664 comments from six platforms

Dữ liệu ViGoEmotions (EACL 2026) có 20.664 bình luận, chia sẵn 16.531 train, 2.066 validation, 2.067 test. Có 28 nhãn, trung bình mỗi bình luận 1,91 nhãn và có 1.485 tổ hợp nhãn khác nhau — nên đây thật sự là bài toán đa nhãn. Nguồn suy ra từ tiền tố id: khoảng 65% là Facebook (id số, bao gồm UIT-VSMEC), còn lại TikTok, YouTube, X, Reddit, Threads. Về giấy phép: chỉ dùng nghiên cứu phi thương mại, không phân phối lại, nên repo của nhóm chỉ chứa code, id của split, số liệu tổng hợp và xác suất dự đoán — không chứa văn bản gốc.

## Slide 4 — Three head labels dominate; the imbalance is only 4.5×

Phân bố nhãn trên tập train: ba nhãn đầu là amusement, sadness, annoyance, mỗi nhãn khoảng 2.700–2.900 mẫu; nhãn hiếm nhất là relief với 635 mẫu. Tỷ lệ mất cân bằng lớn nhất/nhỏ nhất chỉ 4,5 lần — nhẹ hơn nhiều so với GoEmotions tiếng Anh. Quy tắc head/mid/tail được cố định trước khi đánh giá bất kỳ mô hình nào: chia tam phân vị theo số mẫu dương trên TRAIN, được 9 head, 10 mid, 9 tail. Quy tắc này dùng lại cho Final.

## Slide 5 — Two labels per comment is the norm, not the exception

Label cardinality: gần một nửa bình luận có đúng 2 nhãn, 1/3 có 1 nhãn, và khoảng 20% có từ 3 nhãn trở lên; chỉ 3,5% là neutral thuần. Các cặp đồng xuất hiện nhiều nhất là disappointment + sadness (904 lần), anger + annoyance, sadness + grief. Cặp fear + nervousness có hệ số tương quan phi cao nhất (0,41). Đặc biệt 83% bình luận grief cũng là sadness — tức là các nhãn có cấu trúc phân cấp/phụ thuộc mạnh. Theo quy định tái lập, mọi thống kê đồng xuất hiện chỉ tính trên TRAIN.

## Slide 6 — Emotions cluster into families

Ma trận tương quan phi giữa 28 nhãn trên train cho thấy cảm xúc tạo thành các "họ": họ buồn (sadness, grief, disappointment), họ lo âu (fear, nervousness), họ tích cực (excitement, joy, love, optimism, caring, gratitude, relief) và họ thù địch (anger, annoyance, disapproval, disgust). Amusement tương quan âm với sadness. Điều này gợi ý rằng lỗi của mô hình nhiều khả năng là nhầm lẫn TRONG cùng một họ — nhóm sẽ kiểm chứng ở phần phân tích lỗi — và là động lực cho hướng mô hình hóa phụ thuộc nhãn ở Final (RQ2).

## Slide 7 — Short, noisy text: a third uses teencode, a quarter uses emoji

Văn bản rất ngắn: trung vị 11 từ, 47 ký tự; 99% bình luận dưới 84 từ. Sau khi tokenize bằng PhoBERT, p99 là 85 subword nên chọn max_len 192 chỉ cắt 0,01% mẫu. Về đặc điểm phi chuẩn: 31,7% bình luận có teencode (ko, đc, vs, j…), 24,9% có emoji, 11,5% có emoticon kiểu ":))", "=))", 5% có từ tục. Emoji gắn chặt với cảm xúc: gần một nửa bình luận love có emoji. Vì vậy nhóm KHÔNG xóa emoji: với ViSoBERT giữ nguyên (vocab có emoji), với PhoBERT chuyển emoji thành mô tả tiếng Việt.

## Slide 8 — One fixed split for every model, audited for stratification

Về split: nhóm giữ nguyên split chính thức 80/10/10 do tác giả phát hành, để kết quả so sánh trực tiếp được với paper. Nhưng đề yêu cầu split phân tầng đa nhãn, nên nhóm kiểm định: đo độ lệch phân phối nhãn (LD, theo Sechidis 2011) của split chính thức, so với split ngẫu nhiên và split iterative stratification cùng seed. Kết quả: split chính thức chỉ tốt ngang split ngẫu nhiên (0,0035 vs 0,0044), còn iterative tốt hơn khoảng 9 lần. Nhóm vẫn dùng split chính thức, nhưng lưu sẵn split iterative để kiểm tra độ bền ở Final. Kỷ luật: train để fit; validation để early stopping, chọn siêu tham số và ngưỡng; test chỉ chấm một lần.

## Slide 9 — Normalise the noise, keep the emotion signal

Pipeline tiền xử lý gồm 7 bước, mỗi bước bật/tắt được trong config. Điểm đáng chú ý: (1) chuẩn hóa Unicode NFC — code gốc của tác giả thiếu bước này nên khi thay teencode bằng regex, văn bản dạng tổ hợp (NFD) bị hỏng, ví dụ "cảm" thành "cảmày"; (3) chuẩn hóa vị trí dấu thanh theo bảng của VinAI, đúng quy ước dữ liệu pretrain của PhoBERT; (5) emoji: giữ nguyên hoặc dịch sang tiếng Việt, dùng tên emoji tiếng Việt của Unicode CLDR làm dự phòng; (7) tách từ bằng VnCoreNLP RDRSegmenter — đúng bộ tách từ PhoBERT dùng khi pretrain, trong khi paper dùng pyvi. Bảng dưới: mỗi mô hình dùng cấu hình phù hợp với tokenizer của nó.

## Slide 10 — From sparse n-grams to Vietnamese transformers

Ba mô hình bắt buộc: M1 là TF-IDF kết hợp n-gram từ và n-gram ký tự — n-gram ký tự giúp chịu được teencode và lỗi chính tả — với 28 bộ Logistic Regression độc lập; tham số C và class_weight chọn trên validation. M2 là BiLSTM 2 tầng học từ đầu, pooling attention + max. M3 là PhoBERT-base-v2 với đầu sigmoid độc lập và loss BCE đúng như đề. Để có kết quả tốt nhất, nhóm thêm ViSoBERT (mô hình tốt nhất trong paper, pretrain trên mạng xã hội) và ViSoBERT-14GB, rồi ensemble bằng trung bình xác suất. Các kỹ thuật chung: khởi tạo bias theo tần suất nhãn, layer-wise learning-rate decay, dynamic padding, và chọn epoch tốt nhất theo Macro-F1 trên validation.

## Slide 11 — One evaluation module, one threshold policy, for all models

Mọi mô hình đi qua cùng một module đánh giá, cùng split, cùng định dạng. Metric bắt buộc: Macro-F1, Micro-F1, F1 từng nhãn cho cả 28 nhãn, Hamming loss, LRAP và mAP (không phụ thuộc ngưỡng), số tham số và thời gian train. Bảng chính dùng ngưỡng cố định 0,5 cho mọi nhãn. Xác suất liên tục được lưu trước khi nhị phân hóa, nên có thể thử ngưỡng khác mà không cần train lại: ngưỡng toàn cục và ngưỡng theo từng nhãn đều được tối ưu CHỈ trên validation, đóng băng, rồi áp dụng một lần lên test.

## Slide 12 — PhoBERT reaches 0.629 Macro-F1; the ensemble reaches 0.643

Kết quả chính trên test với ngưỡng 0,5, đủ mọi metric bắt buộc. TF-IDF + LR đạt 0,542 Macro-F1, bất ngờ mạnh hơn BiLSTM (0,512) — vì n-gram ký tự chịu tốt teencode, còn BiLSTM học embedding từ đầu với dữ liệu nhỏ. PhoBERT, mô hình bắt buộc, đạt 0,629 Macro-F1 và 0,646 Micro-F1, hơn BiLSTM 11,7 điểm; Hamming loss thấp nhất trong ba mô hình bắt buộc. ViSoBERT và ViSoBERT-14GB đạt 0,613 và 0,620. Ensemble ba transformer (trung bình xác suất, không train lại) đạt 0,643; nếu dùng một ngưỡng toàn cục 0,32 tối ưu trên validation thì đạt 0,653 Macro-F1 và 0,669 Micro-F1. Thời gian train đo trên MacBook M1 Pro; số tham số đếm toàn bộ tham số của mạng.

## Slide 13 — PhoBERT alone beats the paper's best model

So sánh với các baseline đã công bố, cùng split chính thức. Số của paper lấy từ notebook chính thức của tác giả, ngưỡng 0,5. PhoBERT của paper đạt 0,563 (giữ emoji) và 0,593 (đổi emoji sang chữ); mô hình tốt nhất paper là ViSoBERT 0,615. PhoBERT của nhóm đạt 0,629 — một mô hình đơn đã vượt mô hình tốt nhất của paper, và hơn công thức PhoBERT của paper 3,6 điểm. ViSoBERT của nhóm tái lập được kết quả paper (0,613 so với 0,615). Ensemble đạt 0,643 với ngưỡng 0,5 và 0,653 với ngưỡng tune trên validation, tức hơn paper 3,8 điểm. Nhóm liệt kê những điểm khác biệt trong công thức, nhưng lưu ý trung thực: chưa làm ablation từng yếu tố nên chưa khẳng định yếu tố nào đóng góp bao nhiêu — đây là việc cho Final.

## Slide 14 — All three models agree on the easy and the hard labels

Câu hỏi 1: nhãn nào có F1 cao nhất và thấp nhất. Cả ba mô hình đồng thuận: dễ nhất là embarrassment (0,88), gratitude (0,86), amusement — những nhãn có từ khóa rất rõ như "cảm ơn", "ngại quá", "haha". Khó nhất là disapproval (0,36), neutral (0,42), realization, confusion, disgust — đây là các nhãn mơ hồ về ngữ nghĩa, phụ thuộc ngữ cảnh, và trùng với nhóm nhãn yếu nhất trong paper. PhoBERT cao hơn hai baseline ở gần như mọi nhãn; khoảng cách lớn nhất ở realization, neutral, disapproval, admiration — những nhãn cần hiểu ngữ cảnh thay vì từ khóa.

## Slide 15 — Frequency barely predicts F1: difficulty is semantic

Câu hỏi 2: hiệu năng tương quan thế nào với tần suất nhãn trên train. Tương quan dương nhưng yếu và KHÔNG có ý nghĩa thống kê: Spearman khoảng 0,2, p lớn hơn 0,2 cho cả ba mô hình. Tail thấp hơn head khoảng 5 điểm F1, nhưng vì mất cân bằng chỉ 4,5 lần nên tần suất không phải yếu tố chính. Bằng chứng: disapproval có gần 1.000 mẫu train mà F1 thấp nhất; annoyance là nhãn head (2.662 mẫu) nhưng chỉ đạt 0,62; trong khi remorse, grief thuộc tail vẫn trên 0,7. Nhóm kết luận độ khó chủ yếu đến từ ngữ nghĩa mơ hồ và sự chồng lấn giữa các nhãn cùng họ — lưu ý đây là tương quan, không khẳng định nhân quả.

## Slide 16 — PhoBERT reproduces the pair structure; the baselines distort it

Câu hỏi 3: các cặp đồng xuất hiện nhiều nhất (chọn từ TRAIN) là disappointment + sadness, anger + annoyance, sadness + grief. Mô hình có giữ được các phụ thuộc này không? Nhóm so sánh số lần mỗi cặp xuất hiện trong nhãn thật với trong dự đoán trên test. PhoBERT bảo toàn tốt nhất: tương quan Pearson giữa số cặp thật và dự đoán là 0,972, và trung bình dự đoán 1,83 nhãn/câu, gần với thực tế 1,91. TF-IDF với class_weight balanced dự đoán quá nhiều nhãn (2,30/câu) nên sinh gần gấp đôi số cặp, nhất là trong họ thù địch. BiLSTM thì ngược lại, dự đoán quá ít (1,58/câu) và bỏ sót cặp. Lưu ý: cả ba mô hình đều độc lập theo nhãn — phụ thuộc chỉ được học gián tiếp qua biểu diễn chung, đây là động lực cho RQ2 ở Final.

## Slide 17 — Largest gains on emoticon, long and teencode text; smallest on emoji

Câu hỏi 4: PhoBERT vượt BiLSTM nhiều nhất ở kiểu văn bản nào. Các nhóm được định nghĩa bằng quy tắc cố định trong code (src/vigo/analysis/text_styles.py) trước khi xem kết quả mô hình. PhoBERT hơn BiLSTM trên mọi nhóm, và khoảng tin cậy bootstrap đều dương. Chênh lệch lớn nhất ở bình luận có emoticon (+0,19 Macro-F1), bình luận dài (+0,16) và teencode (+0,13): đây là những trường hợp cần hiểu ngữ cảnh. Nhỏ nhất ở bình luận có emoji (+0,06): emoji tự nó đã là tín hiệu cảm xúc mạnh, BiLSTM học trực tiếp emoji như một token, trong khi PhoBERT chỉ thấy mô tả tiếng Việt của emoji. Điều này gợi ý dùng encoder có emoji trong vocab như ViSoBERT.

## Slide 18 — 0.5 is close to optimal on average, but not for each label

Câu hỏi 5: độ nhạy với ngưỡng mặc định 0,5. Quét ngưỡng toàn cục trên validation: đường cong của PhoBERT khá phẳng trong khoảng 0,25–0,5 rồi giảm nhanh sau 0,6. Nghĩa là ở mức trung bình, 0,5 đã gần tối ưu. TF-IDF (có class_weight balanced) tối ưu ở 0,6; BiLSTM ở 0,31. Nhưng khi tối ưu từng nhãn, ngưỡng tốt nhất của PhoBERT dao động từ 0,17 đến 0,69 — một ngưỡng chung rõ ràng không phù hợp với mọi nhãn. Tuy nhiên, tối ưu từng nhãn tăng 2,3 điểm trên validation nhưng chỉ 0,4 điểm trên test: có dấu hiệu overfit ngưỡng. Đây chính là câu hỏi nghiên cứu cho Final.

## Slide 19 — Four recurring failure modes

Phân tích lỗi PhoBERT trên test, ngưỡng 0,5. Heatmap: hàng là nhãn thật bị bỏ sót, cột là nhãn bị dự đoán sai trong cùng bình luận. Bốn kiểu lỗi lặp lại: (1) nhầm trong cùng một họ cảm xúc — annoyance bị đoán thành sadness hoặc disappointment, disgust/annoyance/anger lẫn nhau; khớp với ma trận tương quan ở phần EDA. (2) nhãn ngầm, phụ thuộc ngữ cảnh: disapproval bị bỏ sót 69%, neutral 61%, realization 60% — realization hay bị đọc thành neutral. (3) mô hình bị kéo về khoảng 2 nhãn: câu có 1 nhãn thì bị thêm 0,76 nhãn sai, câu có từ 4 nhãn thì bỏ sót gần 1,9 nhãn. (4) bình luận dài: chỉ 20% khớp hoàn toàn so với 45% ở câu ngắn. Code trích ra 97 trường hợp để audit thủ công (25 false negative, 25 false positive, 15 câu nhiều nhãn, 32 theo kiểu văn bản); file công khai chỉ chứa id và nhãn, bản có văn bản để riêng, không public.

## Slide 20 — RQ1: Is a single uniform threshold (τ = 0.5) optimal across all 27 emotions under class imbalance?

Câu hỏi nghiên cứu đề xuất cho Final là RQ1 của đề: một ngưỡng chung 0,5 có tối ưu cho cả 27 cảm xúc không. Bằng chứng từ midterm: ngưỡng tối ưu theo từng nhãn trải từ 0,17 đến 0,69. Hạ ngưỡng giúp các câu có nhiều nhãn (Micro-F1 cho |Y| từ 4 trở lên tăng từ 0,67 lên 0,75) nhưng lại hại các câu một nhãn (giảm từ 0,57 xuống 0,53). Lợi ích của tối ưu từng nhãn trên validation bị co lại trên test. Thiết kế Final: dùng mạng tốt nhất của midterm, không train lại backbone, so sánh ba chiến lược ngưỡng fit trên validation, phân tích theo head/mid/tail và theo số nhãn, báo cáo khoảng chênh validation–test; chạy 3 seed với khoảng tin cậy bootstrap, thêm biến thể co ngưỡng từng nhãn về ngưỡng chung, và kiểm tra trên split iterative. Xác suất dự đoán đã lưu sẵn đúng schema của Final.

## Slide 21 — Four take-aways from the Midterm

Tổng kết bốn ý chính. Một: nhóm có các baseline mạnh và hợp lệ — PhoBERT 0,629, ensemble 0,653 so với 0,615 của mô hình tốt nhất đã công bố, cùng split, test được niêm phong. Hai: tần suất nhãn giải thích rất ít; nhãn khó là những nhãn ngầm, mơ hồ. Ba: ngữ cảnh pretrain giúp nhiều nhất ở văn bản nhiễu như emoticon, bình luận dài, teencode; lỗi còn lại chủ yếu là nhầm trong cùng họ cảm xúc. Bốn: một ngưỡng chung không phù hợp với mọi nhãn, nhưng tối ưu từng nhãn dễ overfit — đó là câu hỏi nghiên cứu cho Final. Toàn bộ kết quả tái lập được bằng một lệnh hoặc notebook Colab. Cảm ơn thầy/cô và các bạn đã lắng nghe.
