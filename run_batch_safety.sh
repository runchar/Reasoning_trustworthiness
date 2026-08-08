#!/usr/bin/env bash
# 批量运行 HarmfulBench/安全评测（基于 safety/SafetyBench_lmdeploy.py）

# export HF_TOKEN="你的 HuggingFace token"
# export LLAMA_GUARD_PATH="meta-llama/Llama-Guard-3-8B"
# export MODELS_CSV="Qwen/Qwen3-8B,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# bash run_batch_safety.sh

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
# Set HF_TOKEN or HUGGINGFACE_TOKEN in the environment when access is required.

# 将数据路径与模型列表直接写在这里即可，无需命令行参数。
# 可选环境变量：
# OUTPUT_DIR(默认 output)
# MAX_SAMPLES(默认 500)
# LLAMA_GUARD_PATH(默认 meta-llama/Llama-Guard-3-8B)
# HF_TOKEN / HUGGINGFACE_TOKEN (可选，用于下载 Llama Guard)
# TP(主模型 tp，默认 2)
# GUARD_TP(Guard tp，默认 1，建议 1-2)
# SESSION_LEN(主模型会话长度，默认 32768)
# GUARD_SESSION_LEN(Guard 会话长度，默认 4096)
# GUARD_MAX_RETRIES(Llama Guard 不可解析时重试次数，默认 2，范围 1-3)
# MODEL_DEVICES(主模型设备，如 0,1)
# GUARD_DEVICES(Guard 设备，如 2 或 2,3)
# 说明: MODEL_DEVICES / GUARD_DEVICES 是相对于 CUDA_VISIBLE_DEVICES 的本地索引
# MODELS_CSV(逗号分隔模型列表，设置后覆盖默认 MODELS)
# SKIP_VALIDATE=1 可跳过字段校验
DATA_PATH="${DATA_PATH:-./dataset/HarmBench400.json}"
# DEFAULT_MODELS=(
#   "Qwen/Qwen3-8B"
#   "Vinnnf/Thinkless-1.5B-RL-DeepScaleR"
#   "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#   "/data2/yanbowang/reasoning/models/qwen3_8b_dpo"
#   "/data2/yanbowang/reasoning/models/qwen3-8b-tokenskip_v2"
#   "/data2/yanbowang/reasoning/models/DeepSeek-R1-0528-Qwen3-8B-TIES-origin-Merged"
#   "/data2/yanbowang/reasoning/models/DeepSeek-R1-0528-Qwen3-8B-TIES-Merged"
#   "/data2/yanbowang/reasoning/models/DeepSeek-R1-0528-Qwen3-8B-TA-Merged"
#   "/data2/yanbowang/reasoning/models/DeepSeek-R1-0528-Qwen3-8B-DARE-TIES-Merged"
#   "/data2/yanbowang/reasoning/models/DeepSeek-R1-0528-Qwen3-8B-average-Merged"
# )
# DEFAULT_MODELS=(
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# "Zigeng/R1-VeriThinker-7B"
# "agentica-org/DeepScaleR-1.5B-Preview"
# "l3lab/L1-Qwen-1.5B-Exact"
# "l3lab/L1-Qwen-1.5B-Max"
# "l3lab/L1-Qwen-7B-Exact"
# "l3lab/L1-Qwen-7B-Max"
# "l3lab/L1-Qwen3-8B-Exact"
# "l3lab/L1-Qwen3-8B-Max"
# )
DEFAULT_MODELS=(
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)
# DEFAULT_MODELS=(
#   "agentica-org/DeepScaleR-1.5B-Preview"
#   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#   "Qwen/Qwen3-8B"
# )
  
if [[ -n "${MODELS_CSV:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
LLAMA_GUARD_PATH="${LLAMA_GUARD_PATH:-meta-llama/Llama-Guard-3-8B}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
TP=4
GUARD_TP=2
SESSION_LEN="${SESSION_LEN:-32768}"
GUARD_SESSION_LEN="${GUARD_SESSION_LEN:-4096}"
GUARD_MAX_RETRIES="${GUARD_MAX_RETRIES:-2}"
MODEL_DEVICES="${MODEL_DEVICES:-}"
GUARD_DEVICES="${GUARD_DEVICES:-}"

for i in "${!MODELS[@]}"; do
  MODELS[$i]="${MODELS[$i]#"${MODELS[$i]%%[![:space:]]*}"}"
  MODELS[$i]="${MODELS[$i]%"${MODELS[$i]##*[![:space:]]}"}"
done

if [[ "${#MODELS[@]}" -eq 0 || -z "${MODELS[0]}" ]]; then
  echo "错误: MODELS 为空，请设置 MODELS_CSV 或修改默认模型列表"
  exit 1
fi

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
  python tools/validate_benchmark_dataset.py \
    --task safety \
    --data-path "${DATA_PATH}"
fi

TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTPUT_DIR"

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL")
  OUT="$OUTPUT_DIR/safety_${NAME}_${TS}.json"
  echo "==> safety: $MODEL"
  python safety/SafetyBench_lmdeploy.py \
    --model_path "$MODEL" \
    --data_path "$DATA_PATH" \
    --output_path "$OUT" \
    --llama_guard_path "$LLAMA_GUARD_PATH" \
    --hf_token "$HF_TOKEN" \
    --tp "$TP" \
    --guard_tp "$GUARD_TP" \
    --session_len "$SESSION_LEN" \
    --guard_session_len "$GUARD_SESSION_LEN" \
    --guard_max_retries "$GUARD_MAX_RETRIES" \
    --model_devices "$MODEL_DEVICES" \
    --guard_devices "$GUARD_DEVICES" \
    --max_samples "$MAX_SAMPLES"
  echo "saved: $OUT"
done

echo "全部模型评测完成，结果位于 ${OUTPUT_DIR}"
