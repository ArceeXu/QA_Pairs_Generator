import sys
sys.path.insert(0,"./../")

import os
import json
import time
from typing import List, Dict
from utils import QwenGenerator

class EntityAligner:
    """
    使用 Qwen 大模型对 Claim 实体与知识图谱中的 Nodes 实体集合进行统一对齐
    """
    def __init__(self, api_key: str = None, model_name: str = None):
        self.generator = QwenGenerator(api_key=api_key, model_name=model_name)

    def align_entities(self, claim_entities: List[str], kg_entities: List[str]) -> Dict[str, str]:
        """
        对比两组实体，输出标准的对齐字典 {"Claim实体": "KG中的对应真实实体"}
        """
        if not claim_entities or not kg_entities:
            return {}

        role = "You are a helpful assistant for aligning entities from a claim with entities in a knowledge graph."
        prompt = f"""Your task is to align each entity in the 【Claim Entity List】 with the corresponding node in the 【Knowledge Graph Node Entity List】 that refers to the same subject.
If they are the same name, abbreviation and full name, translation differences, singular and plural differences, or synonyms, please establish a mapping. If there is no corresponding entity in the knowledge graph, do not output it in the final JSON.
Claim Entity List:{json.dumps(claim_entities, ensure_ascii=False)}
Knowledge Graph Node Entity List:{json.dumps(kg_entities, ensure_ascii=False)}
Please strictly return a standard pure JSON object format dictionary, where the Key is the Claim entity and the Value is the aligned knowledge graph node entity. Do not return any Markdown markers, and do not write any additional text.
Expected Output Format:
{{
    "Claim Entity 1": "Aligned KG Entity A",
    "Claim Entity 2": "Aligned KG Entity B",
}}
"""
        response_text = self.generator._call_qwen(role, prompt).strip()
        
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        try:
            alignment_dict = json.loads(response_text)
            if isinstance(alignment_dict, dict):
                return alignment_dict
        except Exception:
            print(f"解析对齐JSON失败，网络原返回: {response_text}")
        return {}

    def process_alignment_file(self, input_step2_path: str, output_path: str):
        if not os.path.exists(input_step2_path):
            print(f"未找到输入文件: {input_step2_path}")
            return

        with open(input_step2_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line.strip()) for line in f if line.strip()]

        print(f"开始对 {len(samples)} 个样本进行两端实体对齐...")

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for sample in samples:
                claim_ents = sample.get("claim_entities", [])
                kg_ents = sample.get("all_entities", [])
                
                alignment_map = self.align_entities(claim_ents, kg_ents)
                sample["entity_alignment"] = alignment_map
                
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                print(f"样本 {sample.get('sample_id')} 完成对齐，成功映射 {len(alignment_map)} 个实体")

if __name__ == "__main__":
    aligner = EntityAligner(model_name='qwen-flash-character-2026-02-26')
    aligner.process_alignment_file("../../TMP/wcep_kg_4/step2_output.jsonl", "../../TMP/wcep_kg_4/step3_output.jsonl")