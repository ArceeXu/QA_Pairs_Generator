import sys
sys.path.insert(0,"./../")

import json
from graph_retriever import GraphRetriever
from fact_checker import FactChecker
from qa_generator import QAGenerator

def run_pipeline(step3_output_file: str, final_output_file: str):
    checker = FactChecker(model_name="qwen-flash-character-2026-02-26")
    qa_gen = QAGenerator(model_name="qwen-flash-character-2026-02-26")
    
    with open(step3_output_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]
        
    print(f"开始对 {len(samples)} 个对齐样本生成核查报告与QA对...")
    
    with open(final_output_file, 'w', encoding='utf-8') as out_f:
        for sample in samples:
            all_triplets = sample.get("all_triplets", [])
            kg_index = sample.get("sample_entity_index", {})
            claim_text = sample.get("claim_text", "")
            
            # 获取已经完成对齐的实体列表
            aligned_map = sample.get("entity_alignment", {})
            aligned_entities = list(aligned_map.values())
            
            # a. 检索图谱三元组
            retriever = GraphRetriever(all_triplets, kg_index)
            retrieved_triplets = retriever.retrieve_by_entities(aligned_entities)
            sample["retrieved_triplets"] = retrieved_triplets
            
            # b. 真实性核查判断
            check_res = checker.check_veracity(claim_text, retrieved_triplets)
            sample["fact_check_result"] = check_res
            
            # c. 针对性问答对生成
            qa_pairs = qa_gen.generate_qa_pairs(claim_text, check_res)
            sample["generated_qa_pairs"] = qa_pairs
            
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"样本 {sample['sample_id']} 事实核查完毕: {check_res['verdict']}，生成 QA 对: {len(qa_pairs)} 个")

if __name__ == "__main__":
    # 配置输入与输出路径
    STEP3_OUT = "../../TMP/wcep_kg_4/step3_output.jsonl"
    STEP4_OUT = "../../TMP/wcep_kg_4/step4_output.jsonl"
    
    # 执行流水线
    run_pipeline(step3_output_file=STEP3_OUT, final_output_file=STEP4_OUT)