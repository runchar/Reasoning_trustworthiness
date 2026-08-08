export CUDA_VISIBLE_DEVICES=4,5

python qwen3_eval.py \
    --model-path /shared/ssd/yanbowang/reasoning/models/qwen3_8b_dpo \
    --dataset gsm8k math500 \
    --tp 2 --backend turbomind \
    --output-dir outputs/qwen3-dpo \
    --enable-thinking \
    --max-new-tokens 8164 \
    --temperature 0.0

python qwen3_eval.py \
    --model-path Qwen/Qwen3-8B \
    --dataset gsm8k math500 \
    --tp 2 --backend turbomind \
    --output-dir outputs/qwen3 \
    --enable-thinking \
    --max-new-tokens 8164 \
    --temperature 0.0