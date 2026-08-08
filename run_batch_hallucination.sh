#!/usr/bin/env bash
set -euo pipefail

# 将数据路径与模型列表直接写在这里即可，无需命令行参数。
# SKIP_VALIDATE=1 可跳过字段校验
DATA_PATH="${DATA_PATH:-dataset/FaithEval_unanswerable.json}"
MODELS=(
  "/path/to/modelA"
)

OUTPUT_DIR=${OUTPUT_DIR:-output}
TS=$(date +%Y%m%d-%H%M%S)

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
  python tools/validate_benchmark_dataset.py \
    --task hallucination \
    --data-path "${DATA_PATH}"
fi

mkdir -p "$OUTPUT_DIR"

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL")
  OUT="$OUTPUT_DIR/hallu_${NAME}_${TS}.json"
  echo "==> hallucination: $MODEL"
  python hallucination/HallucinationBench_lmdeploy.py \
    --model_path "$MODEL" \
    --data_path "$DATA_PATH" \
    --output_path "$OUT"
  echo "saved: $OUT"
done
