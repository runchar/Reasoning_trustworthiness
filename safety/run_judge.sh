#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
set -euo pipefail

# ===== 默认参数 =====
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"   
MODEL_NAME=""
HOST="127.0.0.1"
PORT="24444"
TP=4
BACKEND="turbomind"
SESSION_LEN=32768
CACHE_MAX_ENTRY_COUNT=0.4

# ===== 解析 --model =====
for arg in "$@"; do
  case $arg in
    --model=*)
      MODEL="${arg#*=}"
      ;;
  esac
done

if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="$(basename "$MODEL")"
fi

# ===== 启动 =====
lmdeploy serve api_server "$MODEL" \
  --server-port "$PORT" \
  --backend "$BACKEND" \
  --tp "$TP" \
  --session-len "$SESSION_LEN" \
  --cache-max-entry-count "$CACHE_MAX_ENTRY_COUNT" \
  --tool-call-parser qwen \
  --model-name "$MODEL_NAME" 
