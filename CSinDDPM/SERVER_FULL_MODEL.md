# 完整模型服务器运行说明

先阅读 README.md 的 Data 小节，并确认其中三个 TIFF 的使用与再分发权限。所有正式运行都使用新的输出目录，不覆盖既有实验结果。

## 预检

```bash
conda activate csinddpm
python -m pytest -q
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python scripts/verify_best_3data.py --stage data --output revision_outputs/full_model_data_preflight.json
```

## 正式完整模型基线训练（未自动启动）

```bash
TRAIN_OUT="revision_outputs/full_model_baseline_3data"
TRAIN_LOG="server_logs/full_model_baseline_3data"
if [ -e "$TRAIN_OUT" ] || [ -e "$TRAIN_LOG" ]; then
  echo "STOP: 输出或日志已存在；请检查，不覆盖。"
else
  mkdir -p "$TRAIN_LOG"
  nohup env CUDA_VISIBLE_DEVICES=0 \
    EXPERIMENT_VARIANT=A_full USE_CONDITION=true \
    SEEDS=2026 TRAIN_STEPS=60000 SAVE_INTERVAL=5000 \
    VALIDATION_TRIALS=16 VALIDATION_SEED=314159 \
    BEST_CHECKPOINT_METRIC=total_loss \
    DIRTY_PROBABILITY=0.30 DIRTY_MIN_FRAC=0.10 DIRTY_MAX_FRAC=0.30 \
    OUTPUT_ROOT="$TRAIN_OUT" LOG_ROOT="$TRAIN_LOG/per_dataset" \
    bash scripts/train_best_3data.sh > "$TRAIN_LOG/launcher.log" 2>&1 &
  echo "$!" | tee "$TRAIN_LOG/launcher.pid"
fi
```

检查 `tail -n 60 -f server_logs/full_model_baseline_3data/launcher.log`。
三类全部完成并出现 `verification_training.json` 的 PASS 后，再采样。`nohup` 不保证服务器重启后继续；发生中断应先核对现有 checkpoint，再显式使用脚本的 `RESUME_PARTIAL=1`，且所有参数必须保持相同。

## 最佳普通模型采样

```bash
TRAIN_OUT="revision_outputs/full_model_baseline_3data"
SAMPLE_OUT="revision_outputs/full_model_samples_seed1001_1010_3data"
SAMPLE_LOG="server_logs/full_model_samples_seed1001_1010_3data"
if [ -e "$SAMPLE_OUT" ] || [ -e "$SAMPLE_LOG" ]; then
  echo "STOP: 输出或日志已存在；请检查，不覆盖。"
else
  mkdir -p "$SAMPLE_LOG"
  nohup env CUDA_VISIBLE_DEVICES=0 \
    EXPERIMENT_VARIANT=A_full USE_CONDITION=true SEEDS=2026 \
    CHECKPOINT_VARIANT=model NUM_SAMPLES=10 SAMPLING_SEED=1001 \
    TRAIN_STEPS=60000 SAVE_INTERVAL=5000 \
    VALIDATION_TRIALS=16 VALIDATION_SEED=314159 \
    BEST_CHECKPOINT_METRIC=total_loss \
    DIRTY_PROBABILITY=0.30 DIRTY_MIN_FRAC=0.10 DIRTY_MAX_FRAC=0.30 \
    TRAIN_ROOT="$TRAIN_OUT" SAMPLE_ROOT="$SAMPLE_OUT" \
    LOG_ROOT="$SAMPLE_LOG/per_dataset" \
    bash scripts/sample_best_3data.sh > "$SAMPLE_LOG/launcher.log" 2>&1 &
  echo "$!" | tee "$SAMPLE_LOG/launcher.pid"
fi
```

每个样本种子为 1001–1010。采样无样本级续跑功能，部分失败时先保留原输出并检查原因，不要覆盖已有样本。

## 评价入口示例（砂岩）

评价必须针对全部固定种子；更换材料时同时更换 training_tiff、sample_dir、target_porosity 和 output_dir。

```bash
EVAL_OUT="revision_outputs/full_model_evaluation_seed1001_1010_3data/sandstone"
if [ -e "$EVAL_OUT" ]; then
  echo "STOP: 评价输出已存在。"
else
  python scripts/evaluate_samples.py \
    --training_tiff data/sandstone_64_uint8/min_0_0_0.tif \
    --sample_dir revision_outputs/full_model_samples_seed1001_1010_3data/sandstone/train_seed_2026/model \
    --output_dir "$EVAL_OUT" \
    --target_porosity 0.18186569213867188 \
    --pattern '*_binary.tif' --binary_threshold 127.5 --pore_is_high true \
    --connectivity 6 --mpc_definition pending --max_shift 2 \
    --training_patches 512 --generated_patches 128
fi
```

MPC 定义未经确认时保持 pending。后续敏感性实验另行冻结参数范围和输出名称，不要直接修改这个基线目录里的旧结果。
