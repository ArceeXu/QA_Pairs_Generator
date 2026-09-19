import os
import json
import gzip 
import time
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

import sys
sys.path.insert(0,"./../")
from utils import QwenGenerator

class KGSchema:
    """知识图谱模式 - 提供严格约束的 Prompt 骨架"""
    def get_prompt_schema(self, text_content: str) -> str:
        return f"""You are a professional Knowledge Graph extraction specialist. 
Your task is to extract ALL semantic nodes and relationships from the provided text according to strict typing rules.

## Extraction Schema Rules:
1. **Node Types (Must map to one of these types):**
   - ENTITY: PERSON, ORGANIZATION, LOCATION, CONCEPT, OBJECT
   - EVENT: ACTION, STATE_CHANGE, PROCESS, PHENOMENON
   - TIME: TIME_POINT, TIME_INTERVAL, DATE

2. **Relationship Types (Follow a TWO-TIER rule):**
   - **TIER 1 - STRICT (You MUST exactly use these exact names if the meaning fits, do not reinvent):**
     * TEMPORAL: BEFORE, AFTER, DURING, SIMULTANEOUS, HAPPENS_AT
     * CAUSAL: CAUSES, PREVENTS, ENABLES, TRIGGERS, LEADS_TO
     * CONTRADICTORY: CONTRADICTS, INCOMPATIBLE_WITH
   - **TIER 2 - FREE:** For any other semantic relationships, you can create descriptive relation names freely, but they MUST be written in UPPER_SNAKE_CASE (e.g., MEMBER_OF, WORKS_FOR, PARTICIPATED_IN).

## CRITICAL OUTPUT FORMAT:
- You MUST return ONLY a valid JSON object. 
- ABSOLUTELY DO NOT USE any markdown code blocks, backticks (like ```json or ```).
- Output must start with {{ and end with }}. No conversational filler text or explanations.

## Expected JSON Structure:
{{
  "nodes": ["Unique Node Name or Token","EVENT/ENTITY/TIME"],
  "relationships": ["source_node_name","source_node_type","RELATION_TYPE_IN_UPPER_CASE","target_node_name","target_node_type"]
}}

## Text to Extract From:
"{text_content}"
"""

class ArticleKGExtractor:
    """
    使用原生 Qwen 接口进行图谱抽取，带有多层类型解包防御与断点续传
    """
    def __init__(self, api_key: str = None, model_name: str = None, text: str = ""):
        self.generator = QwenGenerator(api_key=api_key, model_name=model_name)
        self.role = "You are a professional Knowledge Graph extraction specialist."  
        self.prompt = KGSchema().get_prompt_schema(text)

    def extract_graph_from_text(self, text:str) -> Dict[str, List]:
        """将清洗和 JSON 解析管道隔离，确保崩溃时提供安全降级"""
        # 为每篇文章动态构建 prompt 并调用模型
        prompt = KGSchema().get_prompt_schema(text)
        raw_response = self.generator._call_qwen(self.role, prompt)

        # 强转为字符串后再进行纯文本清洗
        response_text = str(raw_response).strip()

        # 若返回为空或为工具的空列表占位符，直接返回空图谱
        if not response_text or response_text == "[]":
            return {"nodes": [], "relationships": []}
        
        # 清洗大模型习惯性携带的 Markdown 代码大衣
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        try:
            graph_data = json.loads(response_text)

            if isinstance(graph_data, list):
                print("模型返回了列表结构，无法解析为图谱。返回空结果。", graph_data[:1])
                return {"nodes": [], "relationships": []}

            if isinstance(graph_data, dict) and "nodes" in graph_data and "relationships" in graph_data:
                print(graph_data)
                return graph_data
        except Exception as e:
            print(f"图谱 JSON 反序列化失败。文本片段: {response_text[:150]}... | 异常: {str(e)}")

        return {"nodes": [], "relationships": []}

    def process_wcep_pipeline(self, src_jsonl: str, dst_dir: str, 
                              start_idx: int = 0, end_idx: int = None, max_articles_per_sample: int = None, max_reconnect: int = 3):
        """主流水线：具备断点挂起、追加固化和控制流统计"""
        pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(dst_dir, "wcep_graph_results.jsonl")

        # 1. 兼容读取原始或压缩包文件
        samples = []
        print(f"正在读取 WCEP 数据源: {src_jsonl} ...")
        if src_jsonl.endswith('.gz'):
            with gzip.open(src_jsonl, "rt", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
        else:
            with open(src_jsonl, "r", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
        
        actual_end = end_idx if end_idx is not None else len(samples)
        sliced_samples = samples[start_idx:actual_end]
        print(f"载入总样本数: {len(samples)}，切片处理区间: [{start_idx} : {actual_end}]，共计 {len(sliced_samples)} 个")

        # 2. 检查断点，跳过已持久化样本
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        processed_ids.add(json.loads(line.strip()).get("sample_id"))
            print(f"检测到断点：已有 {len(processed_ids)} 个样本处理完，将自动跳过。")

        # 3. 核心双层迭代处理
        with open(output_file, 'a', encoding='utf-8') as out_f:  # 💡 使用追加模式 'a'
            for sample in tqdm(sliced_samples, desc="Overall Progress"):
                sample_id = sample.get("id") or sample.get("sample_id")
                if sample_id in processed_ids:
                    continue

                raw_articles = sample.get("articles", [])
                extracted_articles = []
                
                max_article = 0
                # 嵌套处理每个样本下的多篇文章
                for art in tqdm(raw_articles, desc=f"Sample {sample_id}", leave=False):
                    art_text = art.get("text", "").strip()
                    if not art_text:
                        continue
                    
                    # 抽取图谱
                    graph_res = self.extract_graph_from_text(art_text)
                    
                    # 对齐你的转换结构：提取原文章元数据，并注入抽取出的 graph 结果
                    extracted_articles.append({
                        "graph": graph_res,  # 💡 内部结构为 {"nodes": [...], "relationships": [...]}
                        "article_id": art.get("id"),
                        "article_title": art.get("title") or art.get("article_title", ""),
                        "article_text": art_text,
                        "extraction_timestamp": time.time()
                    })
                    max_article += 1
                    if max_article >= max_articles_per_sample:
                        break

                final_sample_package = {
                    "sample_id": sample_id,
                    "sample_summary": sample.get("summary") or sample.get("sample_summary", ""),
                    "sample_category": sample.get("category") or sample.get("sample_category", ""),
                    "num_articles_processed": len(extracted_articles),
                    "articles": extracted_articles
                }
                
                out_f.write(json.dumps(final_sample_package, ensure_ascii=False) + "\n")
                out_f.flush()  # 确保强制写入硬盘
                print(f"样本 {sample_id} 抽取完毕 -> 成功抽取文章数: {len(extracted_articles)}")

        print(f"提取结束！数据完整固化保存在: {output_file}")

if __name__ == "__main__":
    extractor = ArticleKGExtractor(model_name='qwen-flash-character-2026-02-26') 
    INPUT_WCEP = "../../raw_data/train.jsonl.gz"  # 若是压缩包可换为 .jsonl.gz
    OUTPUT_DIR = "../../TMP/wcep_kg_4/"
    extractor.process_wcep_pipeline(
        src_jsonl=INPUT_WCEP,
        dst_dir=OUTPUT_DIR,
        start_idx=0,
        end_idx=1,
        max_articles_per_sample=5
    )