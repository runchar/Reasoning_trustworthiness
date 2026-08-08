MODEL_PATH="/data2/yanbowang/reasoning/models/DeepSeek-R1-0528-Qwen3-8B-DARE-TIES-Merged"      # 模型路径
BASE_MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B-DARE-TIES-Merged"   #文件夹内区分
METHOD_TAG="Merged"        #文件夹标签

# MODEL_TAG="${BASE_MODEL_NAME}_${METHOD_TAG}"
MODEL_TAG="${BASE_MODEL_NAME}"

export HF_HOME="${HF_HOME:-/shared/ssd/yanbowang/huggingface}"
export TRANSFORMERS_VERBOSITY=error
export CUDA_VISIBLE_DEVICES=6,7

AGENTHARM_DATA="dataset/AgentHarm396.json"
FAITHEVAL_DATA="dataset/FaithEval_unanswerable.json"
MMLU_PROX_DATA="dataset/MMLU-PROX.json"

OUT_ROOT="output"


#############################################
# 传入一个期望路径，若已存在，则自动加 _v1 / _v2 ... 后缀
#############################################
ensure_unique_file() {
  local target="$1"

  if [[ ! -e "$target" ]]; then
    echo "$target"
    return
  fi

  local base="${target%.*}"
  local ext="${target##*.}"
  local i=1
  local candidate

  while true; do
    candidate="${base}_v${i}.${ext}"
    if [[ ! -e "$candidate" ]]; then
      echo "$candidate"
      return
    fi
    i=$((i+1))
  done
}

#############################################
# HallucinationBench (FaithEval)
#############################################
run_hallucination() {
  echo "==== Running HallucinationBench (FaithEval) for ${MODEL_TAG} ===="

  local out_dir="${OUT_ROOT}/hallucination/faith_eval/"
  mkdir -p "$out_dir"

  local raw_out="${out_dir}/${MODEL_TAG}_hallucination_results_faith_eval_lmdeploy.json"
  local out_path
  out_path=$(ensure_unique_file "$raw_out")

  python -m hallucination.HallucinationBench_lmdeploy \
    --model_path "${MODEL_PATH}" \
    --data_path "${FAITHEVAL_DATA}" \
    --output_path "${out_path}"

  echo "HallucinationBench done. Result: ${out_path}"
}

#############################################
# MultilingualBench (MMLU-PROX)
#############################################
run_multilingual() {
  echo "==== Running MultilingualBench (MMLU-PROX) for ${MODEL_TAG} ===="

  local out_dir="${OUT_ROOT}/multilingual/"
  mkdir -p "$out_dir"

  local raw_out="${out_dir}/${MODEL_TAG}_multilingual_results_filtered_lmdeploy.json"
  local out_path
  out_path=$(ensure_unique_file "$raw_out")

  python -m multilingual.MultilingualBench_lmdeploy \
    --model_path "${MODEL_PATH}" \
    --data_path "${MMLU_PROX_DATA}" \
    --output_path "${out_path}" \
    --samples_per_language 0

  echo "MultilingualBench done. Result: ${out_path}"
}

echo "Model path:       ${MODEL_PATH}"
echo "Base model name:  ${BASE_MODEL_NAME}"
echo "Method tag:       ${METHOD_TAG}"
echo "Model tag:        ${MODEL_TAG}"
echo

run_hallucination
# run_multilingual

echo "==== All benchmarks finished for ${MODEL_TAG} ===="