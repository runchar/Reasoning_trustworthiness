export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=/shared/ssd/yanbowang/huggingface

### thinkless
# python hallucination/HallucinationBench_lmdeploy.py \
#     --model_path agentica-org/DeepScaleR-1.5B-Preview \
#     --data_path dataset/FaithEval_unanswerable.json \
#     --output_path output/hallucination/faith_eval/thinkless/agentica-org_DeepScaleR-1.5B-Preview_thinkless_hallucination_results_faith_eval_lmdeploy.json

# python hallucination/HallucinationBench_lmdeploy.py \
#     --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --data_path dataset/FaithEval_unanswerable.json \
#     --output_path output/hallucination/faith_eval/thinkless/DeepSeek-R1-Distill-Qwen-7B_thinkless_hallucination_results_faith_eval_lmdeploy.json 

python hallucination/HallucinationBench_lmdeploy.py \
    --model_path /data2/yanbowang/reasoning/models/qwen3_8b_tokenskip \
    --data_path dataset/FaithEval_unanswerable.json \
    --output_path output/hallucination/faith_eval/tokenskip/Qwen3-8B_tokenskip_hallucination_results_faith_eval_lmdeploy.json

exit 0

### BeConcise 
# python hallucination/HallucinationBench_lmdeploy.py \
#     --model_path agentica-org/DeepScaleR-1.5B-Preview \
#     --data_path dataset/FaithEval_unanswerable.json \
#     --output_path output/hallucination/faith_eval/BeConcise/agentica-org_DeepScaleR-1.5B-Preview_thinkless_hallucination_results_faith_eval_lmdeploy.json

# python hallucination/HallucinationBench_lmdeploy.py \
#     --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --data_path dataset/FaithEval_unanswerable.json \
#     --output_path output/hallucination/faith_eval/BeConcise/DeepSeek-R1-Distill-Qwen-7B_thinkless_hallucination_results_faith_eval_lmdeploy.json

# python hallucination/HallucinationBench_lmdeploy.py \
#     --model_path Qwen/Qwen3-8B \
#     --data_path dataset/FaithEval_unanswerable.json \
#     --output_path output/hallucination/faith_eval/BeConcise/Qwen3-8B_thinkless_hallucination_results_faith_eval_lmdeploy.json
# exit 0


### origin
# python hallucination/HallucinationBench_lmdeploy.py \
#     --model_path agentica-org/DeepScaleR-1.5B-Preview \
#     --data_path dataset/FaithEval_unanswerable.json \
#     --output_path output/hallucination/faith_eval/agentica-org_DeepScaleR-1.5B-Preview_thinkless_hallucination_results_faith_eval_lmdeploy.json

# python hallucination/HallucinationBench_lmdeploy.py \
#     --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --data_path dataset/FaithEval_unanswerable.json \
#     --output_path output/hallucination/faith_eval/DeepSeek-R1-Distill-Qwen-7B_thinkless_hallucination_results_faith_eval_lmdeploy.json 

# python hallucination/HallucinationBench_lmdeploy.py \
#     --model_path Qwen/Qwen3-8B \
#     --data_path dataset/FaithEval_unanswerable.json \
#     --output_path output/hallucination/faith_eval/Qwen3-8B_thinkless_hallucination_results_faith_eval_lmdeploy.json


### verithinker
python hallucination/HallucinationBench_lmdeploy.py \
    --model_path Zigeng/R1-VeriThinker-7B \
    --data_path dataset/FaithEval_unanswerable.json \
    --output_path output/hallucination/faith_eval/verithinker/Zigeng-R1-VeriThinker-7B_verithinker_hallucination_results_faith_eval_lmdeploy.json

### l1
python hallucination/HallucinationBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen-1.5B-Exact \
    --data_path dataset/FaithEval_unanswerable.json \
    --output_path output/hallucination/faith_eval/l1/L1-Qwen-1.5B-Exact_hallucination_results_faith_eval_lmdeploy.json

python hallucination/HallucinationBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen-1.5B-Max \
    --data_path dataset/FaithEval_unanswerable.json \
    --output_path output/hallucination/faith_eval/l1/L1-Qwen-1.5B-Max_hallucination_results_faith_eval_lmdeploy.json

python hallucination/HallucinationBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen-7B-Exact \
    --data_path dataset/FaithEval_unanswerable.json \
    --output_path output/hallucination/faith_eval/l1/L1-Qwen-7B-Exact_hallucination_results_faith_eval_lmdeploy.json

python hallucination/HallucinationBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen-7B-Max \
    --data_path dataset/FaithEval_unanswerable.json \
    --output_path output/hallucination/faith_eval/l1/L1-Qwen-7B-Max_hallucination_results_faith_eval_lmdeploy.json

python hallucination/HallucinationBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen3-8B-Exact \
    --data_path dataset/FaithEval_unanswerable.json \
    --output_path output/hallucination/faith_eval/l1/L1-Qwen3-8B-Exact_hallucination_results_faith_eval_lmdeploy.json

python hallucination/HallucinationBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen3-8B-Max \
    --data_path dataset/FaithEval_unanswerable.json \
    --output_path output/hallucination/faith_eval/l1/L1-Qwen3-8B-Max_hallucination_results_faith_eval_lmdeploy.json