import argparse
import json
import logging
import os
import random
import re
import statistics
import unicodedata
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from multilingual.token_stats import aggregate_token_stats

from lmdeploy import ChatTemplateConfig, GenerationConfig, TurbomindEngineConfig, pipeline
from transformers import AutoTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    overall_accuracy: float
    macro_accuracy: float
    per_language_accuracy: Dict[str, float]
    generated_tokens_mean: float
    generated_tokens_median: float
    thinking_tokens_mean: float
    thinking_tokens_median: float


class MultilingualBenchLmdeploy:
    def __init__(self, model_path: str):
        logger.info(f"Initialising model {model_path}")
        backend_config = TurbomindEngineConfig(session_len=32768, tp=2, enable_thinking=True, repetition_penalty=1.2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        try:
            token_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            self.thinking_end_token_id = token_ids[0] if token_ids else None
            if self.thinking_end_token_id is not None:
                logger.info(f"Tokenizer loaded. Thinking end token id: {self.thinking_end_token_id}")
            else:
                logger.info("Tokenizer loaded. No thinking end token detected.")
        except Exception as exc:
            logger.warning(f"Failed to detect thinking end token: {exc}")
            self.thinking_end_token_id = None

        self.pipe = pipeline(
            model_path=model_path,
            backend_config=backend_config,
            chat_template_config=ChatTemplateConfig(model_path=model_path, model_name="deepseek-r1")
        )
        logger.info("Model pipeline ready")

    def load_dataset(
        self,
        data_path: str,
        samples_per_language: Optional[int] = None,
        sample_seed: Optional[int] = None
    ) -> List[Dict]:
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        total_samples = len(data)
        logger.info(f"Loaded {total_samples} samples")

        if samples_per_language is not None and samples_per_language > 0:
            logger.info(
                "Balancing dataset to %d samples per language%s",
                samples_per_language,
                f" using seed {sample_seed}" if sample_seed is not None else ""
            )

            grouped: Dict[str, List[Dict]] = defaultdict(list)
            for item in data:
                lang = item.get("language") or "Unknown"
                grouped[lang].append(item)

            rng = random.Random(sample_seed) if sample_seed is not None else None
            balanced: List[Dict] = []
            for lang, items in grouped.items():
                target = min(samples_per_language, len(items))
                if target < samples_per_language:
                    logger.warning(
                        "Language %s only has %d samples; using all available.",
                        lang,
                        len(items)
                    )

                if rng is not None and len(items) > target:
                    indices = list(range(len(items)))
                    rng.shuffle(indices)
                    selected_indices = sorted(indices[:target])
                    balanced.extend(items[idx] for idx in selected_indices)
                else:
                    balanced.extend(items[:target])

            if rng is not None:
                rng.shuffle(balanced)
            data = balanced

        logger.info("Using %d samples for evaluation", len(data))
        return data

    def format_prompt(self, sample: Dict) -> str:
        language = sample.get("language", "Unknown")
        question = sample.get("question", "")
        context = sample.get("context") or sample.get("passage") or sample.get("article")
        choices = sample.get("choices") or []

        if choices:
            choices_text = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))

            INSTRUCTIONS = {
                "Afrikaans": "Jy los 'n veeltalige meerkeuse-eksamen op. Lees die vraag en kies die beste opsie. Antwoord deur die opsieletter (A-J) in [[]] in te sluit, gevolg deur 'n kort motivering in dieselfde taal waar moontlik.",
                "Arabic": "أنت تحل اختبارًا متعدد اللغات بأسئلة اختيار من متعدد. اقرأ السؤال واختر أفضل خيار. أجب بوضع حرف الخيار (A-J) داخل [[]]، متبوعًا بتبرير موجز باللغة نفسها متى أمكن.",
                "Bengali": "আপনি একটি বহুভাষিক বহু নির্বাচনী পরীক্ষা সমাধান করছেন। প্রশ্নটি পড়ুন এবং সেরা বিকল্পটি নির্বাচন করুন। সম্ভব হলে একই ভাষায় সংক্ষিপ্ত যুক্তিসহ উত্তর দেওয়ার সময় বিকল্পের অক্ষরটি (A-J) [[]] এর মধ্যে রাখুন.",
                "Chinese": "你正在解答一个多语言选择题。请阅读问题并选择最佳选项。请将最终的选项字母（A-J）用 [[]] 框选，然后尽可能用与问题相同的语言附上简短的理由。",
                "Czech": "Řešíte vícejazyčný test s výběrem z možností. Přečtěte si otázku a vyberte nejlepší možnost. Odpovězte tak, že písmeno zvolené možnosti (A-J) uzavřete do [[]], a pokud možno přidejte stručné odůvodnění ve stejném jazyce.",
                "English": "You are solving a multilingual multiple-choice exam. Read the question and select the best option. Reply by enclosing the option letter (A-J) in [[]], followed by a brief justification in the same language when possible.",
                "French": "Vous résolvez un examen multilingue à choix multiples. Lisez la question et sélectionnez la meilleure option. Répondez en encadrant la lettre de l’option (A-J) dans [[]], suivie d’une brève justification dans la même langue lorsque c’est possible.",
                "German": "Du bearbeitest eine mehrsprachige Multiple-Choice-Prüfung. Lies die Frage und wähle die beste Option. Antworte, indem du den Buchstaben der Option (A-J) in [[]] setzt und nach Möglichkeit eine kurze Begründung in derselben Sprache gibst.",
                "Hindi": "आप एक बहुभाषी बहुविकल्पीय परीक्षा हल कर रहे हैं। प्रश्न पढ़ें और सबसे अच्छा विकल्प चुनें। उत्तर देते समय विकल्प का अक्षर (A-J) को [[]] में रखें और संभव हो तो उसी भाषा में संक्षिप्त कारण दें।",
                "Hungarian": "Egy többnyelvű feleletválasztós vizsgát oldasz meg. Olvasd el a kérdést, és válaszd ki a legjobb lehetőséget. Válaszolj úgy, hogy a választott opció betűjét (A-J) [[]] közé zárod, majd lehetőség szerint adj rövid indoklást ugyanazon a nyelven.",
                "Indonesian": "Anda sedang mengerjakan ujian pilihan ganda multibahasa. Bacalah pertanyaan dan pilih opsi terbaik. Balas dengan menuliskan huruf opsi (A-J) di dalam [[]], diikuti penjelasan singkat dalam bahasa yang sama jika memungkinkan.",
                "Italian": "Stai risolvendo un esame multilingue a scelta multipla. Leggi la domanda e seleziona l’opzione migliore. Rispondi racchiudendo la lettera dell’opzione (A-J) in [[]], seguita da una breve giustificazione nella stessa lingua quando possibile.",
                "Japanese": "あなたは多言語の選択式試験を解いています。質問を読み、最適な選択肢を選んでください。回答の際は、選択肢の文字（A-J）を [[]] で囲み、可能な限り同じ言語で簡潔な理由を述べてください。",
                "Korean": "당신은 다국어 객관식 시험을 풀고 있습니다. 질문을 읽고 가장 좋은 선택지를 고르세요. 가능한 경우 같은 언어로 짧은 근거를 덧붙이며 선택지 문자(A-J)를 [[]]로 감싸서 답변하세요.",
                "Marathi": "आपण बहुभाषिक बहुपर्यायी परीक्षा सोडवत आहात. प्रश्न वाचा आणि सर्वोत्तम पर्याय निवडा. उत्तर देताना पर्यायाचे अक्षर (A-J) [[]] मध्ये ठेवा आणि शक्य असल्यास त्याच भाषेत थोडक्यात कारण द्या.",
                "Nepali": "तपाईं बहुभाषिक बहुविकल्पीय परीक्षा समाधान गर्दै हुनुहुन्छ। प्रश्न पढ्नुहोस् र उत्तम विकल्प छनोट गर्नुहोस्। उत्तर दिँदा विकल्पको अक्षर (A-J) [[]] भित्र राख्नुहोस् र सम्भव भएमा एउटै भाषामा छोटो कारण लेख्नुहोस्।",
                "Portuguese": "Você está resolvendo um exame multilíngue de múltipla escolha. Leia a pergunta e selecione a melhor opção. Responda colocando a letra da opção (A-J) entre [[]], seguida de uma breve justificativa no mesmo idioma quando possível.",
                "Russian": "Вы решаете многоязычный экзамен с выбором ответа. Прочитайте вопрос и выберите лучший вариант. Ответьте, заключив букву варианта (A-J) в [[]] и при возможности добавьте краткое обоснование на том же языке.",
                "Serbian": "Rešavate višejezični test sa višestrukim izborom. Pročitajte pitanje i odaberite najbolju opciju. Odgovorite tako što ćete slovo izabrane opcije (A-J) staviti u [[]], a zatim po mogućnosti dodati kratko obrazloženje na istom jeziku.",
                "Spanish": "Estás resolviendo un examen multilingüe de opción múltiple. Lee la pregunta y selecciona la mejor opción. Responde encerrando la letra de la opción (A-J) en [[]], seguida de una breve justificación en el mismo idioma cuando sea posible.",
                "Swahili": "Unashughulikia mtihani wa kuchagua majibu wa lugha nyingi. Soma swali na uchague chaguo bora. Jibu kwa kuweka herufi ya chaguo (A-J) ndani ya [[]], kisha toa sababu fupi kwa lugha hiyo hiyo inapowezekana.",
                "Telugu": "మీరు బహుభాషా బహుఎంపిక పరీక్షను పరిష్కరిస్తున్నారు. ప్రశ్నను చదివి ఉత్తమ ఎంపికను ఎంచుకోండి. సాధ్యమైనప్పుడు అదే భాషలో సంక్షిప్త కారణంతో ఎంపిక అక్షరాన్ని (A-J) [[]] లో పెట్టి సమాధానం ఇవ్వండి.",
                "Thai": "คุณกำลังทำข้อสอบแบบเลือกตอบหลายภาษาหลายตัวเลือก อ่านคำถามและเลือกตัวเลือกที่ดีที่สุด ตอบโดยใส่ตัวอักษรของตัวเลือก (A-J) ไว้ใน [[]] แล้วตามด้วยคำอธิบายสั้น ๆ ในภาษาเดียวกันหากเป็นไปได้.",
                "Ukrainian": "Ви розв’язуєте багатомовний тест з вибором відповіді. Прочитайте запитання та оберіть найкращий варіант. Відповідайте, беручи літеру обраного варіанта (A-J) у [[]] і за можливості додаючи коротке пояснення тією самою мовою.",
                "Urdu": "آپ ایک کثیر اللسانی کثیر الانتخاب امتحان حل کر رہے ہیں۔ سوال پڑھیں اور بہترین اختیار منتخب کریں۔ جواب دیتے وقت اختیار کا حرف (A-J) کو [[]] میں رکھیں اور ممکن ہو تو اسی زبان میں مختصر وجہ بیان کریں.",
                "Vietnamese": "Bạn đang làm một bài thi trắc nghiệm đa ngôn ngữ. Đọc câu hỏi và chọn phương án tốt nhất. Trả lời bằng cách đặt chữ cái của phương án (A-J) trong [[]], sau đó kèm một lời giải thích ngắn bằng cùng ngôn ngữ nếu có thể.",
                "Wolof": "Yaa ngiy solal benn tàggat bi ñeel làkk yu bari ak tànnéef yu bari. Jàng laaj bi te tànn topp bi gëna baax. Tuur arafu tànn bi (A-J) ci [[]] te su fekkee ne yëm, yokk benn year ci làkk bii.",
                "Yoruba": "O n yanju idanwo àṣàyàn ọ̀pọ̀ èdè. Ka ìbéèrè naa kí o sì yan aṣayan tó dára jù. Dáhùn nípa fífi lẹ́tà aṣayan (A-J) sínú [[]], lẹ́yìn náà ṣàlàyé kúkúrú ní èdè kan naa bí ó bá ṣeé ṣe.",
                "Zulu": "Usombulula ukuhlolwa kwezilimi eziningi okunokhetho oluningi. Funda umbuzo ukhethe inketho engcono kakhulu. Phendula ngokufaka uhlamvu lwenketho (A-J) ngaphakathi [[]], bese wengeza incazelo emfushane ngolimi olufanayo uma kungenzeka."
            }

            parts = [
                INSTRUCTIONS.get(language, INSTRUCTIONS["English"])
            ]

            if context:
                parts.append(f"Context:\n{context}")
            parts.append(f"Question:\n{question}")
            parts.append(f"Options:\n{choices_text}")
            parts.append("Answer:")
            # parts.append("\n Be Concise")
            return "\n".join(parts)

        pass

    def generate_response(self, prompt: str) -> Tuple[str, List[int]]:
        gen_config = GenerationConfig(
            max_new_tokens=4096,
            temperature=0,
            do_sample=False
        )

        response = self.pipe(prompt, gen_config=gen_config)

        response_text = ""
        token_ids: List[int] = []

        if response is None:
            return response_text, token_ids

        candidate = response[0] if isinstance(response, list) and response else response
        if hasattr(candidate, "text"):
            response_text = candidate.text.strip()
        elif isinstance(candidate, str):
            response_text = candidate.strip()

        if response_text:
            token_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        return response_text, token_ids

    def strip_thinking(self, text: str) -> str:
        if not text:
            return ""
        marker = "</think>"
        if marker in text:
            last_idx = text.rfind(marker)
            text = text[last_idx + len(marker):]
        return text.strip()

    def extract_prediction(self, response: str, choices: List[str]) -> Tuple[Optional[str], Optional[str]]:
        cleaned = self.strip_thinking(response)
        if not cleaned:
            return None, None

        uppercase = cleaned.upper()
        option_pattern = re.compile(r"(?:^|[^A-Z])([A-J])(?:[^A-Z]|$)")
        for match in option_pattern.finditer(uppercase):
            letter = match.group(1)
            idx = ord(letter) - 65
            if 0 <= idx < len(choices):
                return letter, choices[idx]

        normalized = cleaned.casefold()
        for idx, choice in enumerate(choices):
            candidate = choice.strip()
            if not candidate:
                continue
            if candidate.casefold() in normalized:
                letter = chr(65 + idx)
                return letter, choice
        return None, None

    def is_correct(self, predicted_letter: Optional[str], correct_letter: str) -> bool:
        if not predicted_letter:
            return False
        return predicted_letter.strip().upper() == correct_letter.strip().upper()

    def extract_thinking_length_from_tokens(self, token_ids: List[int]) -> int:
        if not token_ids or self.thinking_end_token_id is None:
            return 0
        try:
            index = len(token_ids) - token_ids[::-1].index(self.thinking_end_token_id)
            return index
        except ValueError:
            return 0

    def collect_reference_answers(self, sample: Dict) -> List[str]:
        answers: List[str] = []
        raw_answers = sample.get("answers")
        if isinstance(raw_answers, list):
            for answer in raw_answers:
                if isinstance(answer, dict):
                    text = answer.get("text")
                    if text:
                        answers.append(str(text))
                    aliases = answer.get("aliases")
                    if isinstance(aliases, list):
                        for alias in aliases:
                            if alias:
                                answers.append(str(alias))
                elif isinstance(answer, str):
                    answers.append(answer)

        best_answer = sample.get("best_answer")
        if isinstance(best_answer, str) and best_answer.strip():
            answers.append(best_answer)

        fallback_answer = sample.get("answer")
        if (not sample.get("choices")) and isinstance(fallback_answer, str) and fallback_answer.strip():
            answers.append(fallback_answer)

        seen = set()
        deduped: List[str] = []
        for ans in answers:
            candidate = ans.strip()
            if not candidate:
                continue
            key = self._canonicalize_for_matching(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _canonicalize_for_matching(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _tokenize_for_matching(self, text: str) -> List[str]:
        canonical = self._canonicalize_for_matching(text)
        tokens = canonical.split()
        normalized_tokens: List[str] = []
        for token in tokens:
            if token.isdigit():
                token = token.lstrip("0") or "0"
            normalized_tokens.append(token)
        return normalized_tokens

    def match_freeform_answer(self, response: Optional[str], references: List[str]) -> Tuple[bool, Optional[str]]:
        if not response:
            return False, None
        if not references:
            return False, None

        normalized_response = self._canonicalize_for_matching(response)
        compact_response = normalized_response.replace(" ", "")
        response_tokens = set(self._tokenize_for_matching(response))

        for ref in references:
            normalized_ref = self._canonicalize_for_matching(ref)
            if not normalized_ref:
                continue
            if normalized_ref in normalized_response:
                return True, ref
            compact_ref = normalized_ref.replace(" ", "")
            if compact_ref and compact_ref in compact_response:
                return True, ref
            ref_tokens = self._tokenize_for_matching(ref)
            if ref_tokens and all(token in response_tokens for token in ref_tokens):
                return True, ref
        return False, None

    def evaluate_sample(self, sample: Dict) -> Dict:
        prompt = self.format_prompt(sample)
        response, token_ids = self.generate_response(prompt)

        result: Dict[str, Optional[str]] = {
            "question": sample.get("question"),
            "language": sample.get("language"),
            "response": response,
        }

        choices = sample.get("choices") or []
        if choices:
            predicted_letter, predicted_choice = self.extract_prediction(response, choices)
            correct_letter = (sample.get("answer") or "").strip().upper()
            result.update({
                "choices": choices,
                "correct_letter": correct_letter,
                "predicted_letter": predicted_letter,
                "predicted_choice": predicted_choice,
                "is_correct": self.is_correct(predicted_letter, correct_letter),
            })
        else:
            reference_answers = self.collect_reference_answers(sample)
            predicted_answer = self.strip_thinking(response) or None
            is_match, matched_reference = self.match_freeform_answer(predicted_answer, reference_answers)
            result.update({
                "reference_answers": reference_answers,
                "predicted_answer": predicted_answer,
                "matched_reference": matched_reference,
                "is_correct": is_match,
            })

        if token_ids:
            thinking_length = self.extract_thinking_length_from_tokens(token_ids)
            result["generated_tokens"] = len(token_ids)
            result["thinking_tokens"] = thinking_length

        return result

    def calculate_metrics(self, results: List[Dict]) -> EvaluationMetrics:
        total = len(results)
        if total == 0:
            return EvaluationMetrics(0.0, 0.0, {}, 0.0, 0.0, 0.0, 0.0)

        overall_correct = sum(1 for item in results if item["is_correct"])
        overall_accuracy = overall_correct / total

        per_language_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        for item in results:
            lang = item.get("language") or "Unknown"
            per_language_counts[lang]["total"] += 1
            if item["is_correct"]:
                per_language_counts[lang]["correct"] += 1

        per_language_accuracy = {
            lang: (counts["correct"] / counts["total"]) if counts["total"] else 0.0
            for lang, counts in per_language_counts.items()
        }

        macro_accuracy = statistics.mean(per_language_accuracy.values()) if per_language_accuracy else 0.0

        generated_tokens = [item.get("generated_tokens") for item in results if item.get("generated_tokens") is not None]
        thinking_tokens = [item.get("thinking_tokens") for item in results if item.get("thinking_tokens") is not None]

        gen_stats, think_stats = aggregate_token_stats(generated_tokens, thinking_tokens)

        return EvaluationMetrics(
            overall_accuracy=overall_accuracy,
            macro_accuracy=macro_accuracy,
            per_language_accuracy=per_language_accuracy,
            generated_tokens_mean=float(gen_stats["mean"]),
            generated_tokens_median=float(gen_stats["median"]),
            thinking_tokens_mean=float(think_stats["mean"]),
            thinking_tokens_median=float(think_stats["median"])
        )

    def run_evaluation(
        self,
        data_path: str,
        output_path: str = "multilingual_results_lmdeploy.json",
        samples_per_language: Optional[int] = None,
        sample_seed: Optional[int] = None
    ) -> EvaluationMetrics:
        dataset = self.load_dataset(
            data_path,
            samples_per_language=samples_per_language,
            sample_seed=sample_seed
        )
        results: List[Dict] = []

        logger.info("Starting multilingual evaluation...")
        for sample in tqdm(dataset, desc="Multilingual evaluation"):
            result = self.evaluate_sample(sample)
            results.append(result)

        metrics = self.calculate_metrics(results)

        output = {
            "metrics": {
                "overall_accuracy": float(metrics.overall_accuracy),
                "macro_accuracy": float(metrics.macro_accuracy),
                "per_language_accuracy": {
                    lang: float(acc) for lang, acc in metrics.per_language_accuracy.items()
                },
                "generated_tokens_mean": metrics.generated_tokens_mean,
                "generated_tokens_median": metrics.generated_tokens_median,
                "thinking_tokens_mean": metrics.thinking_tokens_mean,
                "thinking_tokens_median": metrics.thinking_tokens_median
            },
            "details": results
        }

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        logger.info("Evaluation completed")
        logger.info(f"Overall accuracy: {metrics.overall_accuracy:.4f}")
        logger.info(f"Macro language accuracy: {metrics.macro_accuracy:.4f}")
        for lang, acc in sorted(metrics.per_language_accuracy.items()):
            logger.info(f"{lang}: {acc:.4f}")
        logger.info(f"Results saved to: {output_path}")

        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Multilingual benchmark for language models using lmdeploy")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the multilingual dataset")
    parser.add_argument("--output_path", type=str, default="multilingual_results_lmdeploy.json", help="Where to store the evaluation report")
    parser.add_argument("--samples_per_language", type=int, default=0, help="Number of samples to draw per language (0 means all)")
    parser.add_argument("--sample_seed", type=int, default=0, help="Sampling seed to synchronise selections across datasets (0 means deterministic order)")

    args = parser.parse_args()
    samples_per_language = args.samples_per_language if args.samples_per_language > 0 else None
    sample_seed = args.sample_seed if args.sample_seed != 0 else None

    bench = MultilingualBenchLmdeploy(model_path=args.model_path)
    bench.run_evaluation(
        data_path=args.data_path,
        output_path=args.output_path,
        samples_per_language=samples_per_language,
        sample_seed=sample_seed
    )


if __name__ == "__main__":
    main()
