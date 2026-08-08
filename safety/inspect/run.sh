# export INSPECT_LOG_LEVEL=debug
# export INSPECT_TRACE=1


inspect eval inspect_evals/agentharm \
  --model openai-api/lmdeploy_sut/Qwen3-8B  \
  --limit 2000 \
  --log-format json \
  --log-dir logs/Qwen3-8B-Prompt \
  --no-parallel-tool-calls \
  --no-fail-on-error \
  --retry-on-error=2 \
  -T refusal_judge=openai/gpt-4o-mini \
  -T semantic_judge=openai/gpt-4o-mini \
  --system-message "Be concise."



# inspect eval inspect_evals/agentharm \
#   --model openai-api/lmdeploy/Qwen3-8B \
#   --limit 10 \
#   -T refusal_judge=openai-api/glm/glm-4 \
#   -T semantic_judge=openai-api/glm/glm-4