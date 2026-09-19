import sys

sys.path.insert(0, "./../")

import os
import json
from typing import List, Dict

from utils import QwenGenerator


class RegularEvaluator:
    """
    遵循 REGULAR 论文评估框架，利用大模型作为评判官(LLM-as-a-judge)
    对生成的问答对进行 Faithfulness(忠实度), Relevance(相关性), Completeness(完备性) 细粒度打分。
    """

    def __init__(self, api_key: str = None, model_name: str = None):
        self.generator = QwenGenerator(api_key=api_key, model_name=model_name)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            return text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        if text.startswith("```"):
            return text.split("```", 1)[1].rsplit("```", 1)[0].strip()
        return text

    def evaluate_qa_set(self, claim: str, evidence: str, explanation: str, qa_pairs: List[Dict]) -> Dict:
        if not qa_pairs:
            print("警告: 评估的 QA 对列表为空，无法进行评估。")
            return {"avg_faithfulness": 0.0, "avg_relevance": 0.0, "avg_completeness": 0.0, "qa_eval_details": []}

        qa_pairs_text = json.dumps(qa_pairs, ensure_ascii=False, indent=2)

        prompt = f"""You are a meticulous and objective judge for evaluating the quality of question-answer pairs generated based on a claim, its key evidence, and an explanation. Please evaluate each question-answer pair in the following three dimensions:
1. Faithfulness: Does the answer accurately reflect the information in the key evidence and explanation without adding any unsupported information?
2. Relevance: Is the question relevant to the claim and the key evidence? Does the answer directly address the question?
3. Completeness: Does the answer provide a complete response to the question, covering all necessary aspects based on the claim, evidence, and explanation?

Claim:
{claim}

Key evidence:
{evidence}

Explanation:
{explanation}

Question-answer pairs to evaluate:
{qa_pairs_text}

Please provide a score from 0 to 1 for each dimension for every question-answer pair, and also calculate the average score for each dimension across all pairs. Return your evaluation in the following JSON format without any additional explanations or formatting:
{{
    "avg_faithfulness": 0.85,
    "avg_relevance": 0.9,
    "avg_completeness": 0.8,
    "qa_eval_details": [
        {{
            "question": "What is the main claim being evaluated?",
            "answer": "The main claim is that ...",
            "faithfulness": 0.9,
            "relevance": 1.0,
            "completeness": 0.8
        }}
    ]
}}"""

        try:
            res_text = self.generator._call_qwen(
                role="You are a meticulous and objective judge for evaluating the quality of question-answer pairs.",
                prompt=prompt,
            )
            res_text = self._strip_code_fence(res_text)
            return json.loads(res_text)
        except Exception as e:
            print(f"评估执行错误: {str(e)}")
            return {"avg_faithfulness": 0.0, "avg_relevance": 0.0, "avg_completeness": 0.0, "error": str(e)}

    def run_evaluation_pipeline(self, step4_output_file: str, evaluation_report_file: str):
        if not os.path.exists(step4_output_file):
            print(f"未找到数据文件: {step4_output_file}")
            return

        with open(step4_output_file, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]

        print(f"开始基于 REGULAR 框架对 {len(samples)} 个生成的 QA 对进行自动化裁判评估...")

        with open(evaluation_report_file, "w", encoding="utf-8") as out_f:
            for sample in samples:
                claim = sample.get("claim_text", "")
                fc_res = sample.get("fact_check_result", {})
                evidence = fc_res.get("key_evidence", "")
                explanation = fc_res.get("explanation", "")
                qa_pairs = sample.get("generated_qa_pairs", [])

                eval_report = self.evaluate_qa_set(claim, evidence, explanation, qa_pairs)
                sample["regular_evaluation"] = eval_report

                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                print(f"样本 {sample['sample_id']} 评估完成 | 忠实度均分: {eval_report.get('avg_faithfulness')} | 相关性均分: {eval_report.get('avg_relevance')}")


if __name__ == "__main__":
    STEP4_OUT = "../../TMP/wcep_kg_4/step4_output.jsonl"
    EVAL_REPORT_OUT = "../../TMP/wcep_kg_4/regular_report.jsonl"

    evaluator = RegularEvaluator(model_name="qwen-flash-character-2026-02-26")
    evaluator.run_evaluation_pipeline(step4_output_file=STEP4_OUT, evaluation_report_file=EVAL_REPORT_OUT)