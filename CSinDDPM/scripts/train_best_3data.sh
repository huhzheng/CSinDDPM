#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_STEPS="${TRAIN_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-16}"
VALIDATION_SEED="${VALIDATION_SEED:-314159}"
BEST_CHECKPOINT_METRIC="${BEST_CHECKPOINT_METRIC:-total_loss}"
OUTPUT_ROOT="${OUTPUT_ROOT:-revision_outputs/full_model_baseline_3data}"
LOG_ROOT="${LOG_ROOT:-server_logs/full_model_baseline_3data}"
RESUME_PARTIAL="${RESUME_PARTIAL:-0}"
EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-A_full}"
USE_CONDITION="${USE_CONDITION:-true}"
DIRTY_PROBABILITY="${DIRTY_PROBABILITY:-0.30}"
DIRTY_MIN_FRAC="${DIRTY_MIN_FRAC:-0.10}"
DIRTY_MAX_FRAC="${DIRTY_MAX_FRAC:-0.30}"
STRICT_PROTOCOL=1

if [[ "$USE_CONDITION" != "true" ]]; then
  echo "ERROR: this full-model distribution requires USE_CONDITION=true"
  exit 2
fi

read -r -a SEED_LIST <<< "${SEEDS:-2026}"

NAMES=(
  sandstone
  carbonate
  mt_gambier_limestone
)

DATASETS=(
  "data/sandstone_64_uint8/min_0_0_0.tif"
  "data/CarbonateRock/carbonate_min_448_256_384.tif"
  "data/Mt Gambier limestone/mt_gambier_z256_y384_x128.tif"
)

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)'
"$PYTHON_BIN" -m pytest -q

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
"$PYTHON_BIN" scripts/verify_best_3data.py \
  --stage data \
  --output "$OUTPUT_ROOT/data_preflight.json"

for seed in "${SEED_LIST[@]}"; do
  for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    data="${DATASETS[$i]}"
    out="$OUTPUT_ROOT/$name/train_seed_$seed"
    log="$LOG_ROOT/${name}_seed_${seed}.log"

    printf -v final_name 'model%06d.pt' "$TRAIN_STEPS"
    final="$out/$final_name"

    if [[ ! -f "$data" ]]; then
      echo "ERROR: input TIFF not found: $data"
      exit 2
    fi

    if [[ -f "$final" ]] &&
       "$PYTHON_BIN" scripts/resolve_best_checkpoint.py \
         --train-dir "$out" --variant model >/dev/null 2>&1; then
      echo "SKIP completed best-checkpoint run: $out"
      continue
    fi

    resume_args=()
    log_mode="write"
    if [[ -d "$out" ]] &&
       [[ -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      if [[ "$RESUME_PARTIAL" != "1" ]]; then
        echo "ERROR: partial output directory exists: $out"
        echo "Inspect it first, then rerun with RESUME_PARTIAL=1 to resume the latest ordinary checkpoint."
        exit 3
      fi
      shopt -s nullglob
      checkpoints=("$out"/model[0-9]*.pt)
      shopt -u nullglob
      if (( ${#checkpoints[@]} == 0 )); then
        echo "ERROR: no resumable model checkpoint found in $out"
        exit 4
      fi
      mapfile -t checkpoints < <(printf '%s\n' "${checkpoints[@]}" | sort)
      latest="${checkpoints[${#checkpoints[@]}-1]}"
      resume_args=(--resume_checkpoint "$latest")
      log_mode="append"
      echo "RESUME: $latest"
    else
      mkdir -p "$out"
    fi

    echo "START $(date -Is): dataset=$name seed=$seed metric=$BEST_CHECKPOINT_METRIC"
    command=(
      "$PYTHON_BIN" -u demo_train.py
      --experiment_variant "$EXPERIMENT_VARIANT"
      --data_dir "$data"
      --output_dir "$out"
      --seed "$seed"
      --batch_size 1
      --microbatch 1
      --num_workers 0
      --lr 0.0001
      --weight_decay 0.0
      --train_steps "$TRAIN_STEPS"
      --diffusion_steps 1000
      --noise_schedule linear
      --image_size 64
      --num_channels 64
      --num_res_blocks 1
      --channel_mult "1,2,4"
      --attention_resolutions "2"
      --num_head_channels 16
      --learn_sigma true
      --vb_weight 1.0
      --use_condition "$USE_CONDITION"
      --dirty_probability "$DIRTY_PROBABILITY"
      --dirty_min_frac "$DIRTY_MIN_FRAC"
      --dirty_max_frac "$DIRTY_MAX_FRAC"
      --ema_rate 0.9999
      --save_interval "$SAVE_INTERVAL"
      --log_interval "$LOG_INTERVAL"
      --use_checkpoint false
      --use_fp16 false
      --pore_is_high true
      --select_best_checkpoint true
      --validation_trials "$VALIDATION_TRIALS"
      --validation_seed "$VALIDATION_SEED"
      --best_checkpoint_metric "$BEST_CHECKPOINT_METRIC"
      --validate_ema true
      "${resume_args[@]}"
    )
    if [[ "$log_mode" == "append" ]]; then
      "${command[@]}" 2>&1 | tee -a "$log"
    else
      "${command[@]}" 2>&1 | tee "$log"
    fi

    "$PYTHON_BIN" scripts/resolve_best_checkpoint.py \
      --train-dir "$out" --variant model --json | tee -a "$log"
    "$PYTHON_BIN" scripts/resolve_best_checkpoint.py \
      --train-dir "$out" --variant ema_0.9999 --json | tee -a "$log"
    "$PYTHON_BIN" scripts/resolve_best_checkpoint.py \
      --train-dir "$out" --variant overall --json | tee -a "$log"
    echo "DONE $(date -Is): dataset=$name seed=$seed"
  done
done

protocol_args=()
if [[ "$STRICT_PROTOCOL" == "1" ]]; then
  protocol_args=(
    --expected-experiment-variant "$EXPERIMENT_VARIANT"
    --expected-use-condition "$USE_CONDITION"
    --expected-dirty-probability "$DIRTY_PROBABILITY"
    --expected-dirty-min-frac "$DIRTY_MIN_FRAC"
    --expected-dirty-max-frac "$DIRTY_MAX_FRAC"
  )
fi

"$PYTHON_BIN" scripts/verify_best_3data.py \
  --stage training \
  --train-root "$OUTPUT_ROOT" \
  --seeds "${SEED_LIST[@]}" \
  --train-steps "$TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --metric "$BEST_CHECKPOINT_METRIC" \
  --validation-trials "$VALIDATION_TRIALS" \
  --validation-seed "$VALIDATION_SEED" \
  "${protocol_args[@]}" \
  --output "$OUTPUT_ROOT/verification_training.json"
