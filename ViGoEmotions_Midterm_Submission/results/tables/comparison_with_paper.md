| model                                | decoding                  |   test_macro_f1 |   test_micro_f1 | source                                     |
|:-------------------------------------|:--------------------------|----------------:|----------------:|:-------------------------------------------|
| PhoBERT-base-v2 (paper, emoji kept)  | tau=0.5, pos_weight BCE   |          0.5633 |          0.5740 | Hung et al. (EACL 2026) official notebooks |
| PhoBERT-base-v2 (paper, emoji->text) | tau=0.5, pos_weight BCE   |          0.5931 |          0.6042 | Hung et al. (EACL 2026) official notebooks |
| CafeBERT (paper)                     | tau=0.5, pos_weight BCE   |          0.6117 |          0.6247 | Hung et al. (EACL 2026) official notebooks |
| ViSoBERT (paper, best)               | tau=0.5, pos_weight BCE   |          0.6150 |          0.6346 | Hung et al. (EACL 2026) official notebooks |
| TF-IDF + OvR LR (ours)               | tau=0.5                   |          0.5419 |          0.5576 | this work                                  |
| TF-IDF + OvR LR (ours)               | per-label tau (val-tuned) |          0.5395 |          0.5458 | this work                                  |
| BiLSTM (ours)                        | tau=0.5                   |          0.5122 |          0.5330 | this work                                  |
| BiLSTM (ours)                        | per-label tau (val-tuned) |          0.5178 |          0.5336 | this work                                  |
| PhoBERT-base-v2 (ours)               | tau=0.5                   |          0.6292 |          0.6461 | this work                                  |
| PhoBERT-base-v2 (ours)               | per-label tau (val-tuned) |          0.6336 |          0.6474 | this work                                  |
| ViSoBERT (ours)                      | tau=0.5                   |          0.6133 |          0.6316 | this work                                  |
| ViSoBERT (ours)                      | per-label tau (val-tuned) |          0.6125 |          0.6252 | this work                                  |
| ViSoBERT-14GB (ours)                 | tau=0.5                   |          0.6204 |          0.6423 | this work                                  |
| ViSoBERT-14GB (ours)                 | per-label tau (val-tuned) |          0.6227 |          0.6423 | this work                                  |
| Ensemble (3) (ours)                  | tau=0.5                   |          0.6430 |          0.6636 | this work                                  |
| Ensemble (3) (ours)                  | per-label tau (val-tuned) |          0.6488 |          0.6654 | this work                                  |