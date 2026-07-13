# Nano summary model - training status

Auto-published by the nightly training cycle on gpu3.
Model assets ship separately (Git LFS); this file tracks the metrics curve.


## 2026-07-10
- generated total: 498 sft rows (+498 new) — below 1000, no training today

## 2026-07-10
- rows total: 4509 (+4509 trained today, 1 epoch continuation from BASE)
- eval_loss: 1.7825677394866943 | holdout gates: {"n": 30, "format_valid": 29, "empty": 1, "format_valid_rate": 0.967, "empty_rate": 0.033} | MTP acceptance=62.42%
- checkpoint: /home/metricspace/models/nano-daily/ckpt-2026-07-10 (+ full/stripped Q4 gguf)
- ** PROMOTION CANDIDATE ** (best format_valid_rate so far: 0.967) -> run promote_checkpoint.sh /home/metricspace/models/nano-daily/ckpt-2026-07-10

## 2026-07-10
- TRAINING FAILED (rc=1, new=30651) — see train_2026-07-10.log; state not advanced

## 2026-07-10
- rows total: 35160 (+30651 trained today, 1 epoch continuation from /home/metricspace/models/nano-daily/ckpt-2026-07-10)
- eval_loss: 1.5629082918167114 | holdout gates: {"n": 30, "format_valid": 30, "empty": 0, "format_valid_rate": 1.0, "empty_rate": 0.0} | MTP acceptance=46.32%
- checkpoint: /home/metricspace/models/nano-daily/ckpt-2026-07-10 (+ full/stripped Q4 gguf)
- ** PROMOTION CANDIDATE FAILED GATES ** (rate=1.0) - see promote_2026-07-10.log

## 2026-07-11
- rows total: 55274 (+20114 trained today, 1 epoch continuation from /home/metricspace/models/nano-daily/ckpt-2026-07-10)
- eval_loss: n/a | holdout gates: {"n": 30, "format_valid": 30, "empty": 0, "format_valid_rate": 1.0, "empty_rate": 0.0} | MTP acceptance=43.50%
- checkpoint: /home/metricspace/models/nano-daily/ckpt-2026-07-11 (+ full/stripped Q4 gguf)
- ** PROMOTED ** best gated checkpoint staged: release-staging/ckpt-2026-07-11
- auto-release to main: RELEASED: d45e08ad7bb8787ae9b6f56b6915e8b44ac6e13c6b740fdc7bd591249209a72c to main

## 2026-07-12
- SKIPPED: another training holds the train lock

## 2026-07-13
- TRAINING FAILED (rc=1, new=214105) — see train_2026-07-13.log; state not advanced
