import os
import json
from typing import List, Dict
from utils import QwenGenerator

class QAGenerator:
    """
    依据事实核查结果与关键证据，通过Qwen衍生出四种维度的问答对(QA Pairs)
    """
    def __init__(self, api_key: str = None, model_name: str = None):
        self.generator = QwenGenerator(api_key=api_key, model_name=model_name)

    def generate_qa_pairs(self, claim: str, check_result: Dict) -> List[Dict]:
        prompt = f"""You need to generate question-answer pairs based on a claim, its fact-checking verdict, and key evidence. The fact-checking result is provided in the following JSON format:{json.dumps(check_result, ensure_ascii=False)}
Based on the claim, the fact-checking verdict, and the key evidence, please generate question-answer pairs in the following four dimensions:
1. Comprehension Questions: Questions that test understanding of the claim and its context.
2. Evidence-Based Questions: Questions that require using the key evidence to answer.
3. Counterfactual Questions: Questions that explore hypothetical scenarios related to the claim.
4. Critical Thinking Questions: Questions that encourage analysis and evaluation of the claim and evidence.
Please return a JSON array of question-answer pairs, where each pair includes the question, the answer, and the dimension it belongs to. Do not include any additional explanations or formatting.
Expected Output Format:
[
    {
        "question": "What is the main claim being evaluated?",
        "answer": "The main claim is that ...",
        "dimension": "Comprehension"
    },
    {
        "question": "What evidence supports the claim?",
        "answer": "The key evidence supporting the claim is ...",
        "dimension": "Evidence-Based"
    },
    {
        "question": "What if the key evidence was different?",
        "answer": "If the key evidence was different, then ...",
        "dimension": "Counterfactual"
    },
    {
        "question": "How would you evaluate the strength of the claim based on the evidence?",
        "answer": "The strength of the claim can be evaluated as ...",
        "dimension": "Critical Thinking"
    }
"""
        try:
            response = self.generator._call_qwen(role="You are a helpful assistant for generating question-answer pairs.", prompt=prompt)
            res_text = response.strip()
            if res_text.startswith("```json"):
                res_text = res_text.split("```json")[1].split("```")[0].strip()
            elif res_text.startswith("```"):
                res_text = res_text.split("```")[1].split("```")[0].strip()
            return json.loads(res_text)
        except Exception:
            return []