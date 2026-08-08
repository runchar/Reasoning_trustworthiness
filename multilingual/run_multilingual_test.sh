export CUDA_VISIBLE_DEVICES=6,7
export HF_HOME=/shared/ssd/yanbowang/huggingface
export TRANSFORMERS_VERBOSITY=error

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path /data2/yanbowang/reasoning/models/qwen3_8b_tokenskip \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/tokenskip/Qwen3-8B_tokenskip_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

exit 0



### thinkless
python multilingual/MultilingualBench_lmdeploy.py \
    --model_path agentica-org/DeepScaleR-1.5B-Preview \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/thinkless/agentica-org_DeepScaleR-1.5B-Preview_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/thinkless/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path Qwen/Qwen3-8B \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/thinkless/Qwen3-8B_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

exit 0


### Be Concise
python multilingual/MultilingualBench_lmdeploy.py \
    --model_path agentica-org/DeepScaleR-1.5B-Preview \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/BeConcise/agentica-org_DeepScaleR-1.5B-Preview_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/BeConcise/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path Qwen/Qwen3-8B \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/BeConcise/Qwen3-8B_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

exit 0

### origin
# python multilingual/MultilingualBench_lmdeploy.py \
#     --model_path agentica-org/DeepScaleR-1.5B-Preview \
#     --data_path dataset/MMLU-PROX.json \
#     --output_path output/multilingual/agentica-org_DeepScaleR-1.5B-Preview_multilingual_results_filtered_lmdeploy.json \
#     --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path Qwen/Qwen3-8B \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/Qwen3-8B_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

# exit 0

### verithinker
python multilingual/MultilingualBench_lmdeploy.py \
    --model_path Zigeng/R1-VeriThinker-7B \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/verithinker/Zigeng-R1-VeriThinker-7B_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

### l1
python multilingual/MultilingualBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen-1.5B-Exact \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/l1/L1-Qwen-1.5B-Exact_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen-1.5B-Max \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/l1/L1-Qwen-1.5B-Max_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen-7B-Exact \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/l1/L1-Qwen-7B-Exact_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen-7B-Max \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/l1/L1-Qwen-7B-Max_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen3-8B-Exact \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/l1/L1-Qwen3-8B-Exact_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0

python multilingual/MultilingualBench_lmdeploy.py \
    --model_path l3lab/L1-Qwen3-8B-Max \
    --data_path dataset/MMLU-PROX.json \
    --output_path output/multilingual/l1/L1-Qwen3-8B-Max_multilingual_results_filtered_lmdeploy.json \
    --samples_per_language 0
