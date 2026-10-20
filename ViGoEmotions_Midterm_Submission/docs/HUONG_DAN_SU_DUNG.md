# Hướng dẫn sử dụng — ViGoEmotions Midterm

Tài liệu này dành cho nhóm. Tài liệu kỹ thuật đầy đủ bằng tiếng Anh nằm trong `README.md`.

## 1. Trong gói nộp có gì

| Thư mục / file | Nội dung |
|---|---|
| `report/ViGoEmotions_Midterm_Report.docx` (+ `.pdf`) | Báo cáo midterm (Word + PDF), 21 trang, 10 hình, 24 bảng. Số liệu được sinh tự động từ `results/`. |
| `slides/ViGoEmotions_Midterm_Slides.pptx` (+ `.pdf`) | Slide thuyết trình (21 slide, có speaker notes tiếng Việt). Mở bằng PowerPoint/Keynote/Google Slides để chỉnh sửa. |
| `results/ViGoEmotions_Midterm_Results.xlsx` | Workbook Excel gồm 19 sheet, chứa mọi bảng kết quả. Cột Δ và "Best model" là công thức Excel. |
| `results/tables/`, `results/figures/` | Toàn bộ bảng CSV và hình PNG sinh bằng code. |
| `results/predictions/` | Xác suất và dự đoán val/test theo đúng schema của Final. |
| `results/runs/` | Mỗi model có đủ config, env, summary, history và xác suất. |
| `notebooks/` | Notebook 00 (chạy toàn bộ trên Colab) và 01–05 (EDA → model → đánh giá). |
| `src/vigo/` | Package Python: tiền xử lý, model, huấn luyện, đánh giá. |
| `configs/`, `splits/` | Siêu tham số (YAML) và id của split cố định. |
| `docs/` | Hướng dẫn này, checklist yêu cầu midterm, speaker notes tiếng Việt. |

**Không kèm dữ liệu gốc** (`data/`): giấy phép ViGoEmotions cấm phân phối lại. Muốn chạy lại, bạn tải dữ liệu từ Hugging Face `uitnlp/vigoemotions` (gated) hoặc repo `ricardo-tran/ViGoEmotions`, rồi đặt `train.csv`, `val.csv`, `test.csv`, `label_dict.json` vào `data/`.

## 2. Việc cần làm trước khi nộp hoặc thuyết trình

- [ ] Điền `[Team name]`, `[Member 1–3]`, `[Presentation date]` trên slide bìa.
- [ ] Điền `[Team name]`, `[Member …]`, `[Submission date]` trên trang bìa báo cáo.
- [ ] Nếu muốn slide giống hệt bản web: mở artifact deck → **Share › Export → PowerPoint/PDF**, rồi thay vào `slides/`.
- [ ] Hỏi giảng viên xem midterm có cần nộp báo cáo PDF riêng không. Brief của Track 2 không ghi, nhưng student guide có liệt kê.
- [ ] Nếu nộp lên repository public: **không** đưa `data/` và `results/private/` lên (đã có trong `.gitignore`).

## 3. Chạy trên máy cá nhân (macOS / Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
bash scripts/run_all.sh              # toàn bộ: EDA → split → 3 model bắt buộc → ViSoBERT ×2 → ensemble → phân tích
SKIP_EXTRA=1 bash scripts/run_all.sh # chỉ 3 model bắt buộc
```

- Mac M1 Pro: PhoBERT khoảng 5 phút/epoch, toàn bộ pipeline mất khoảng 3–4 giờ. GPU T4 trên Colab nhanh hơn khoảng 2 lần.
- VnCoreNLP (tách từ cho PhoBERT) cần Java. Nếu không có Java, code tự chuyển sang pyvi và in cảnh báo.

Chạy từng bước bằng CLI của package (`python -m vigo -h` để xem danh sách lệnh):

```bash
python -m vigo eda                                    # phân tích dữ liệu
python -m vigo split --make iterative                 # split + kiểm định phân tầng
python -m vigo train-tfidf       --config configs/tfidf_lr.yaml
python -m vigo train-rnn         --config configs/bilstm.yaml
python -m vigo train-transformer --config configs/phobert.yaml
python -m vigo analyze --tfidf tfidf_lr --rnn bilstm --transformer phobert --extra visobert visobert14gb ensemble_3
```

Kiểm tra nhanh xem code chạy được không (chỉ vài bước, không train đủ epoch):

```bash
python -m vigo train-transformer --config configs/phobert.yaml train.epochs=1 train.max_steps=3 run_name=_smoke
```

## Cấu trúc mã nguồn `src/vigo/`

| Package | Nội dung |
|---|---|
| `data/` | đọc dữ liệu và nhãn, split cố định, tiền xử lý tiếng Việt, vocabulary, DataLoader |
| `models/` | TF-IDF + LR, BiLSTM, transformer + sigmoid head, các hàm loss |
| `training/` | vòng lặp train chung, entry point cho từng model, ensemble |
| `evaluation/` | metric, chiến lược ngưỡng, lưu và đọc xác suất của từng run |
| `analysis/` | EDA, nhóm kiểu văn bản (Q4), bảng kết quả, Q1–Q5, phân tích lỗi |
| `utils/` | đường dẫn, config/JSON, seed, device, ghi môi trường |

## 4. Chạy trên Google Colab

1. Upload cả thư mục project (kèm `data/`) lên Google Drive, đặt tên `MyDrive/vigoemotions_midterm/`.
2. Mở `notebooks/00_colab_quickstart.ipynb` → Runtime → Change runtime type → **GPU**.
3. Nếu thư mục ở chỗ khác, sửa `PROJECT_DIR` trong ô đầu tiên, rồi chọn Run all.
4. CafeBERT và PhoBERT-large (tuỳ chọn) cần GPU ≥ 16 GB (L4/A100). Bỏ comment trong notebook 00 để chạy.

## 5. Kết quả chính (test, cùng split với paper)

| Model | Macro-F1 (τ = 0.5) | Micro-F1 |
|---|---|---|
| TF-IDF + OvR LR | 0.542 | 0.558 |
| BiLSTM | 0.512 | 0.533 |
| **PhoBERT-base-v2** | **0.629** | **0.646** |
| ViSoBERT | 0.613 | 0.632 |
| ViSoBERT-14GB | 0.620 | 0.642 |
| **Ensemble 3 transformer** | **0.643** | **0.664** |
| Ensemble + ngưỡng toàn cục tune trên val (τ* = 0.32) | **0.653** | **0.669** |
| *Paper, model tốt nhất (ViSoBERT)* | *0.615* | *0.635* |

Số liệu gốc nằm ở `results/tables/main_results_test.csv` và `threshold_strategies.csv`.

## 6. Những điểm cần nói rõ khi thuyết trình (trung thực)

- Tất cả model chạy **1 seed (42)**. Chênh lệch nhỏ (khoảng 0.5 điểm) chưa đủ để kết luận.
- **Chưa làm ablation** từng khác biệt so với công thức của paper, nên không khẳng định yếu tố nào tạo ra cải thiện.
- Split chính thức chỉ phân tầng ở mức ngẫu nhiên. Split iterative được lưu sẵn để kiểm tra ở Final.
- Learning rate của ViSoBERT-14GB được chỉnh sau khi xem đường cong **validation** của ViSoBERT (không dùng test).
- CafeBERT và PhoBERT-large có config sẵn nhưng **chưa chạy**.
