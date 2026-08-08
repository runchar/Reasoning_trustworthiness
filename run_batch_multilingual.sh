#!/usr/bin/env bash
set -euo pipefail

# 将数据路径与模型列表直接写在这里即可，无需命令行参数。
# 可选环境变量: OUTPUT_DIR(默认 output) SAMPLES_PER_LANG SAMPLE_SEED
# SKIP_VALIDATE=1 可跳过字段校验
DATA_PATH="${DATA_PATH:-dataset/MMLU-PROX.json}"
MODELS=(
  "/path/to/modelA"
)

OUTPUT_DIR=${OUTPUT_DIR:-output}
SAMPLES_PER_LANG=${SAMPLES_PER_LANG:-0}
SAMPLE_SEED=${SAMPLE_SEED:-0}
TS=$(date +%Y%m%d-%H%M%S)

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
  python tools/validate_benchmark_dataset.py \
    --task multilingual \
    --data-path "${DATA_PATH}"
fi

mkdir -p "$OUTPUT_DIR"

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL")
  OUT="$OUTPUT_DIR/multi_${NAME}_${TS}.json"
  echo "==> multilingual: $MODEL"
  python multilingual/MultilingualBench_lmdeploy.py \
    --model_path "$MODEL" \
    --data_path "$DATA_PATH" \
    --output_path "$OUT" \
    --samples_per_language "$SAMPLES_PER_LANG" \
    --sample_seed "$SAMPLE_SEED"
  echo "saved: $OUT"
done
