#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ROOT="${TRAIN_ROOT:-revision_outputs/full_model_baseline_3data}"
SAMPLE_ROOT="${SAMPLE_ROOT:-revision_outputs/full_model_samples_seed1001_1010_3data}"
LOG_ROOT="${LOG_ROOT:-server_logs/full_model_samples_seed1001_1010_3data}"
CHECKPOINT_VARIANT="${CHECKPOINT_VARIANT:-model}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SAMPLING_SEED="${SAMPLING_SEED:-1001}"
TRAIN_STEPS="${TRAIN_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-16}"
VALIDATION_SEED="${VALIDATION_SEED:-314159}"
BEST_CHECKPOINT_METRIC="${BEST_CHECKPOINT_METRIC:-total_loss}"
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

mkdir -p "$SAMPLE_ROOT" "$LOG_ROOT"

training_protocol_args=()
sampling_protocol_args=()
if [[ "$STRICT_PROTOCOL" == "1" ]]; then
  training_protocol_args=(
    --expected-experiment-variant "$EXPERIMENT_VARIANT"
    --expected-use-condition "$USE_CONDITION"
    --expected-dirty-probability "$DIRTY_PROBABILITY"
    --expected-dirty-min-frac "$DIRTY_MIN_FRAC"
    --expected-dirty-max-frac "$DIRTY_MAX_FRAC"
  )
  sampling_protocol_args=(
    --expected-experiment-variant "$EXPERIMENT_VARIANT"
    --expected-use-condition "$USE_CONDITION"
  )
fi

"$PYTHON_BIN" scripts/verify_best_3data.py \
  --stage training \
  --train-root "$TRAIN_ROOT" \
  --seeds "${SEED_LIST[@]}" \
  --train-steps "$TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --metric "$BEST_CHECKPOINT_METRIC" \
  --validation-trials "$VALIDATION_TRIALS" \
  --validation-seed "$VALIDATION_SEED" \
  "${training_protocol_args[@]}" \
  --output "$TRAIN_ROOT/verification_before_sampling.json"

for seed in "${SEED_LIST[@]}"; do
  for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    data="${DATASETS[$i]}"
    train_dir="$TRAIN_ROOT/$name/train_seed_$seed"
    out="$SAMPLE_ROOT/$name/train_seed_$seed/$CHECKPOINT_VARIANT"
    log="$LOG_ROOT/${name}_seed_${seed}_${CHECKPOINT_VARIANT}.log"

    if [[ -d "$out" ]] &&
       [[ -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "ERROR: sampling output already exists: $out"
      echo "This sampler has no sample-level resume. Preserve the partial directory and choose a fresh SAMPLE_ROOT."
      exit 3
    fi
    mkdir -p "$out"

    checkpoint="$("$PYTHON_BIN" scripts/resolve_best_checkpoint.py \
      --train-dir "$train_dir" \
      --variant "$CHECKPOINT_VARIANT" \
      --record-output "$out/checkpoint_selection.json")"
    cp "$train_dir/best_checkpoints.json" "$out/best_checkpoints_manifest.json"
    target_porosity="$("$PYTHON_BIN" -c 'import sys, tifffile; a=tifffile.imread(sys.argv[1]); print(float((a > 0).mean()))' "$data")"

    echo "START $(date -Is): dataset=$name seed=$seed variant=$CHECKPOINT_VARIANT checkpoint=$checkpoint"
    "$PYTHON_BIN" -u demo_sample_num.py \
      --experiment_variant "$EXPERIMENT_VARIANT" \
      --model_path "$checkpoint" \
      --data_dir "$data" \
      --output_dir "$out" \
      --seed "$SAMPLING_SEED" \
      --target_porosity "$target_porosity" \
      --num_samples "$NUM_SAMPLES" \
      --batch_size 1 \
      --use_ddim false \
      --clip_denoised true \
      --progress false \
      --save_raw true \
      --sample_threshold 0.0 \
      --diffusion_steps 1000 \
      --noise_schedule linear \
      --image_size 64 \
      --num_channels 64 \
      --num_res_blocks 1 \
      --channel_mult "1,2,4" \
      --attention_resolutions "2" \
      --num_head_channels 16 \
      --learn_sigma true \
      --vb_weight 1.0 \
      --use_condition "$USE_CONDITION" \
      --use_checkpoint false \
      --use_fp16 false \
      --pore_is_high true \
      2>&1 | tee "$log"
    echo "DONE $(date -Is): dataset=$name seed=$seed variant=$CHECKPOINT_VARIANT"
  done
done

"$PYTHON_BIN" scripts/verify_best_3data.py \
  --stage sampling \
  --sample-root "$SAMPLE_ROOT" \
  --seeds "${SEED_LIST[@]}" \
  --variant "$CHECKPOINT_VARIANT" \
  --samples-per-dataset "$NUM_SAMPLES" \
  --sampling-seed "$SAMPLING_SEED" \
  "${sampling_protocol_args[@]}" \
  --output "$SAMPLE_ROOT/verification_${CHECKPOINT_VARIANT}.json"
