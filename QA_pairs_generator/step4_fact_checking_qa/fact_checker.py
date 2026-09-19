import os
import json
from utils import QwenGenerator

class FactChecker:
    """
    调用通义千问判断 Claim 与 检索出的知识图谱三元组 之间的一致性
    """
    def __init__(self, api_key: str = None, model_name: str = None):
        self.generator = QwenGenerator(api_key=api_key, model_name=model_name)

    def check_veracity(self, claim: str, retrieved_triplets: list) -> dict:
        triplets_str = "\n".join([f"({t[0]}, {t[1]}, {t[2]})" for t in retrieved_triplets])
        
        prompt = f"""You are a helpful assistant for fact-checking a claim against a list of retrieved knowledge graph triplets. Your task is to determine whether the claim is supported, refuted, or if there is not enough information (NEI) based on the provided triplets. 
Claim: "{claim}"
Retrieved Knowledge Graph Triplets:
{triplets_str if triplets_str else "No triplets retrieved."}
Please strictly analyze the claim in relation to the retrieved triplets and provide a verdict of "SUPPORT", "PARTIALLY_SUPPORTED", "PARTIALLY_CONTRADICTED", "CONTRADICTED", or "NEI". 
You must return the expected JSON object without any additional explanations or markdown formatting:
{{
    "verdict": "SUPPORT/PARTIALLY_SUPPORTED/PARTIALLY_CONTRADICTED/CONTRADICTED/NEI",
    "confidence": 0.85,
    "explanation": "A brief explanation of the reasoning behind the verdict.",
    "key_evidence": "The most relevant triplet(s) that support the verdict, if any."
}}
"""
        try:
            response = self.generator._call_qwen(role="You are a helpful assistant for fact-checking.", prompt=prompt)
            res_text = response.strip()
            if res_text.startswith("```json"):
                res_text = res_text.split("```json")[1].split("```")[0].strip()
            elif res_text.startswith("```"):
                res_text = res_text.split("```")[1].split("```")[0].strip()
            return json.loads(res_text)
        except Exception as e:
            return {
                "verdict": "NEI",
                "confidence": 0.0,
                "explanation": f"Fact-checking failed: {str(e)}",
                "key_evidence": ""
            }