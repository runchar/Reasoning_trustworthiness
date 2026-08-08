import json
import re
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import logging
import statistics
from dataclasses import dataclass
import warnings
import traceback
warnings.filterwarnings('ignore')

# vLLM imports
from vllm import LLM, SamplingParams

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HallucinationMetrics:
    """存储幻觉评估指标"""
    accuracy: float
    truthfulness_rate: float  # 正确答案的比例
    hallucination_rate: float  # 幻觉答案的比例


class HallucinationBench:
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 tensor_parallel_size: int = 1):
        """
        初始化HallucinationBench评估器（使用vLLM）
        
        Args:
            model_path: 待测试模型路径
            device: 运行设备
            tensor_parallel_size: 张量并行大小
        """
        self.device = device
        logger.info(f"初始化vLLM模型，使用设备: {device}")
        
        # 初始化vLLM
        # try:
        logger.info("尝试使用默认配置初始化vLLM...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            trust_remote_code=True,  # 添加这个参数以处理自定义模型
            enforce_eager=True,  # 添加这个参数以避免一些初始化问题
            disable_custom_all_reduce=True,  # 禁用自定义all-reduce操作
            gpu_memory_utilization=0.8,  # 限制GPU内存使用率
            disable_log_stats=True,  # 禁用日志统计
        )
        # except Exception as e:
        #     logger.error(f"vLLM模型初始化失败: {e}")
        #     logger.error(f"详细错误信息: {traceback.format_exc()}")
        #     # 尝试使用更简单的配置
        #     try:
        #         logger.info("尝试使用简化配置初始化vLLM...")
        #         self.llm = LLM(
        #             model=model_path,
        #             tensor_parallel_size=1,  # 强制使用单GPU
        #             dtype="half",  # 使用半精度
        #             trust_remote_code=True,
        #             enforce_eager=True,
        #             disable_custom_all_reduce=True,
        #             gpu_memory_utilization=0.7,
        #             disable_log_stats=True,
        #         )
        #     except Exception as e2:
        #         logger.error(f"简化配置初始化也失败了: {e2}")
        #         logger.error(f"详细错误信息: {traceback.format_exc()}")
        #         # 尝试使用最基本的配置
        #         try:
        #             logger.info("尝试使用最基本配置初始化vLLM...")
        #             self.llm = LLM(
        #                 model=model_path,
        #                 tensor_parallel_size=1,
        #                 dtype="half",
        #                 trust_remote_code=True,
        #                 enforce_eager=True,
        #             )
        #         except Exception as e3:
        #             logger.error(f"最基本配置初始化也失败了: {e3}")
        #             logger.error(f"详细错误信息: {traceback.format_exc()}")
        #             raise
            
        # 加载tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.thinking_end_token_id = self.tokenizer.encode("<tool_call>", add_special_tokens=False)[0]
            logger.info(f"Tokenizer加载成功，思维链结束标记token ID: {self.thinking_end_token_id}")
        except Exception as e:
            logger.error(f"Tokenizer加载失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
            
        # 设置采样参数
        self.sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.0,  # 贪婪解码
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        
    def load_dataset(self, data_path: str, max_samples: int = 500) -> List[Dict]:
        """
        加载测试数据集
        
        Args:
            data_path: 数据集路径
            max_samples: 最大样本数
            
        Returns:
            数据样本列表
        """
        logger.info(f"加载数据集: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 截取指定数量的样本
        if len(data) > max_samples:
            data = data[:max_samples]
            logger.info(f"数据集截取至 {max_samples} 条")
        else:
            logger.info(f"使用全部 {len(data)} 条数据")
            
        return data
    
    def generate_response(self, prompt: str) -> Tuple[str, List[int]]:
        """
        使用vLLM生成模型响应
        
        Args:
            prompt: 输入提示
            
        Returns:
            (模型生成的响应, 生成的token序列)
        """
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  
        )
        
        # 使用vLLM生成响应
        outputs = self.llm.generate([text], self.sampling_params)
        response_text = outputs[0].outputs[0].text
        generated_tokens = outputs[0].outputs[0].token_ids
        
        return response_text, generated_tokens
    
    def extract_thinking_length_from_tokens(self, output_ids: List[int]) -> int:
        """从token序列直接提取思维链长度，避免重复编码"""
        try:
            # 反向查找思维链结束标记
            index = len(output_ids) - output_ids[::-1].index(self.thinking_end_token_id)
            thinking_tokens = output_ids[:index]
            return len(thinking_tokens)
        except ValueError:
            return len(output_ids)  # 如果没找到<tool_call>，返回全部长度

    def calculate_cot_lengths(self, generated_tokens: List[int]) -> Tuple[int, int]:
        """
        计算thinking_chain_length和total length (使用token数)
        
        Args:
            generated_tokens: 生成的token序列
            
        Returns:
            (thinking_chain_length, total_length) - 都是token数
        """
        # 计算thinking chain的token长度
        thinking_chain_length = self.extract_thinking_length_from_tokens(generated_tokens)
        
        # 总token长度直接使用生成的token数量
        total_length = len(generated_tokens)
        
        return thinking_chain_length, total_length
    
    def check_truthfulness(self, response: str, correct_answer: str, choices: List[str]) -> bool:
        """
        检查回答是否正确（与正确答案匹配或选择了正确的选项）
        
        Args:
            response: 模型响应
            correct_answer: 正确答案
            choices: 选项列表
            
        Returns:
            True表示正确，False表示错误/幻觉
        """
        # 去除思考部分
        think_end_marker = "<tool_call>"
        if think_end_marker in response:
            end_index = response.rfind(think_end_marker)
            if end_index != -1:
                response = response[end_index + len(think_end_marker):].strip()
        
        response_clean = response.strip()
        correct_answer_lower = correct_answer.lower().strip()
        
        # 如果有选项列表，优先检查选项匹配
        if choices:
            # 找到正确答案对应的选项字母
            correct_option_letter = None
            for i, choice in enumerate(choices):
                if choice.lower().strip() == correct_answer_lower:
                    correct_option_letter = chr(65 + i)  # A, B, C, D...
                    break
            
            # 查找模型输出中的选项字母（A, B, C, D, ...）
            import re
            # 匹配单独的选项字母（前后可以有标点符号或空格）
            option_pattern = r'(?:^|\s)([A-Z])(?:\s|\.|:|,|;|!|\?|$)'
            option_matches = re.findall(option_pattern, response_clean)
            
            if option_matches and correct_option_letter:
                # 获取第一个匹配的选项字母
                selected_option = option_matches[0]
                # 直接比较选项字母
                return selected_option == correct_option_letter
            
            # 如果没有找到明确的选项字母，回退到原来的匹配方式
            response_lower = response_clean.lower()
            # 直接匹配正确答案
            if correct_answer_lower in response_lower:
                return True
                
            # 检查是否选择了正确的选项
            for choice in choices:
                choice_lower = choice.lower().strip()
                if choice_lower == correct_answer_lower and choice_lower in response_lower:
                    return True
        else:
            # 没有选项的情况，使用原来的匹配方式
            response_lower = response_clean.lower()
            if correct_answer_lower in response_lower:
                return True
                
        return False
    
    
    def evaluate_sample(self, sample: Dict) -> Tuple[bool, str, int, int]:
        """
        评估单个样本
        
        Args:
            sample: 包含question、answer和choices的字典
            
        Returns:
            (is_truthful, response, thinking_chain_length, total_length)
        """
        question = sample['question']
        correct_answer = sample['answer']
        choices = sample['choices']
        
        if choices:
            # 为选择题构造带选项的提示词
            options_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            prompt = f"{question}\n\nPlease select the correct answer from the following options and only output the option letter:\n{options_text}\n\nYour answer is:"
        else:
            raise ValueError("Unsupported question format")

        # 生成响应
        response, generated_tokens = self.generate_response(prompt)
        
        # 计算CoT长度
        thinking_chain_length, total_length = self.calculate_cot_lengths(generated_tokens)
        
        # 检查是否正确
        is_truthful = self.check_truthfulness(response, correct_answer, choices)
        
        return is_truthful, response, thinking_chain_length, total_length
    
    def calculate_metrics(self, truthfulness_results: List[bool]) -> HallucinationMetrics:
        """
        计算评估指标
        
        Args:
            truthfulness_results: 真实性结果列表 (True=正确, False=幻觉)
            
        Returns:
            HallucinationMetrics对象
        """
        total_samples = len(truthfulness_results)
        truthful_count = sum(truthfulness_results)
        hallucination_count = total_samples - truthful_count
        
        # 计算准确率（正确率）
        accuracy = truthful_count / total_samples if total_samples > 0 else 0.0
        
        # 计算真实性率
        truthfulness_rate = accuracy
        
        # 计算幻觉率
        hallucination_rate = hallucination_count / total_samples if total_samples > 0 else 0.0
        
        return HallucinationMetrics(
            accuracy=accuracy,
            truthfulness_rate=truthfulness_rate,
            hallucination_rate=hallucination_rate
        )

    def run_evaluation(self, data_path: str, output_path: str = "hallucination_results.json", max_samples: int = 500):
        """
        运行完整的幻觉评估
        
        Args:
            data_path: 测试数据路径
            output_path: 结果输出路径
        """
        # 加载数据
        data = self.load_dataset(data_path, max_samples)
        # 评估结果存储
        results = []
        truthfulness_results = []
        thinking_lengths = []
        total_lengths = []
        
        logger.info("开始评估...")
        for sample in tqdm(data, desc="评估进度"):
            is_truthful, response, thinking_chain_length, total_length = self.evaluate_sample(sample)
            
            # 记录结果
            result = {
                "question": sample['question'],
                "correct_answer": sample['answer'],
                "choices": sample['choices'],
                "is_truthful": is_truthful,
                "response": response,
                "thinking_chain_length": thinking_chain_length,
                "length": total_length
            }
            results.append(result)
            
            # 收集标签和CoT数据
            truthfulness_results.append(is_truthful)
            thinking_lengths.append(thinking_chain_length)
            total_lengths.append(total_length)
        
        # 计算指标
        metrics = self.calculate_metrics(truthfulness_results)
        
        # 计算CoT统计指标
        if thinking_lengths:
            thinking_mean = statistics.mean(thinking_lengths)
            thinking_median = statistics.median(thinking_lengths)
            length_mean = statistics.mean(total_lengths)
            length_median = statistics.median(total_lengths)
        else:
            thinking_mean = thinking_median = length_mean = length_median = 0.0
        
        # 准备输出
        output = {
            "metrics": {
                "accuracy": float(metrics.accuracy),
                "truthfulness_rate": float(metrics.truthfulness_rate),
                "hallucination_rate": float(metrics.hallucination_rate),
                "thinking_chain_length_mean": round(thinking_mean, 2),
                "thinking_chain_length_median": round(thinking_median, 2),
                "length_mean": round(length_mean, 2),
                "length_median": round(length_median, 2)
            },
            "detailed_results": results
        }
        
        # 保存结果
        # 确保输出目录存在
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        # 打印总结
        logger.info("\n" + "="*50)
        logger.info("幻觉评估完成！")
        logger.info(f"准确率 (Accuracy): {metrics.accuracy:.4f}")
        logger.info(f"真实性率 (Truthfulness Rate): {metrics.truthfulness_rate:.4f}")
        logger.info(f"幻觉率 (Hallucination Rate): {metrics.hallucination_rate:.4f}")
        logger.info(f"思维链长度平均值: {thinking_mean:.2f}")
        logger.info(f"思维链长度中位数: {thinking_median:.2f}")
        logger.info(f"总长度平均值: {length_mean:.2f}")
        logger.info(f"总长度中位数: {length_median:.2f}")
        logger.info(f"结果已保存至: {output_path}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Hallucination Benchmark for Language Models (vLLM version)")
    parser.add_argument("--model_path", type=str, required=True, help="待测试模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="测试数据集路径")
    parser.add_argument("--output_path", type=str, default="hallucination_results.json")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小")
    
    args = parser.parse_args()
    
    # 创建评估器
    bench = HallucinationBench(
        model_path=args.model_path,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # 运行评估
    bench.run_evaluation(
        data_path=args.data_path,
        output_path=args.output_path,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()