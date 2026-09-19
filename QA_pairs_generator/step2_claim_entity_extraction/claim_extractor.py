import os
import json
import time
from typing import List, Dict
from utils import QwenGenerator

class ClaimEntityExtractor:
    """
    拼接摘要和标题形成 Claim，并调用 Qwen 提取其中的关键实体
    """
    def __init__(self, api_key: str = None, model_name: str = 'qwen-plus'):
        self.generator = QwenGenerator(api_key=api_key, model_name=model_name)

    def extract_entities(self, claim_text: str) -> List[str]:
        """
        调用 LLM 抽取实体，严格约定返回 JSON 数组
        """
        prompt = f"""Read the following claim text and extract the key entities mentioned in it. An entity can be a person, organization, location, event, or any specific noun that is central to the claim. Please return the entities as a JSON array of strings, without any additional explanations or formatting.
Claim Text: "{claim_text}" 
Example Response: ["Entity1", "Entity2", "Entity3"]"""
        role = "You are a helpful assistant for extracting key entities from a given claim text."
        raw_response = self.generator._call_qwen(role, prompt)
        
        # 强转为字符串后再进行清洗
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
            
        response_text = raw_response.strip()
        
        # 清洗 JSON Markdown 格式
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        try:
            entities = json.loads(response_text)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities]
        except Exception:
            # 基础后备正则兼容或返回空
            print(f"解析 JSON 实体失败，原始返回: {response_text}")
        return []

    def process_converted_file(self, input_converted_path: str, output_path: str):
        """
        对 Step1 的输出结果追加抽取 Claim 实体
        """
        if not os.path.exists(input_converted_path):
            print(f"未找到输入文件: {input_converted_path}")
            return

        results = []
        with open(input_converted_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))

        print(f"开始提取共计 {len(results)} 个样本的 Claim 实体...")
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for sample in results:
                summary = sample.get("sample_summary", "")
                titles = [art.get("article_title", "") for art in sample.get("articles", [])]
                
                # 拼接 Claim 
                claim_text = f"Summary: {summary} | Titles: {' ; '.join(titles)}"
                sample["claim_text"] = claim_text
                
                # 抽取实体
                claim_entities = self.extract_entities(claim_text)
                sample["claim_entities"] = claim_entities
                
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                print(f"样本 {sample.get('sample_id')} 实体提取完成: {len(claim_entities)} 个")

if __name__ == "__main__":
    # 示例运行
    extractor = ClaimEntityExtractor(model_name='qwen-flash-character-2026-02-26')
    extractor.process_converted_file("../../TMP/wcep_kg_4/step1_output.jsonl", "../../TMP/wcep_kg_4/step2_output.jsonl")