import json
import os
from typing import List, Dict
from collections import defaultdict
from pathlib import Path

class TripletConverter:
    """
    将 WCEP 的 nodes/relationships 转换为标准三元组格式
    保留文章来源信息，构建实体索引
    """
    def __init__(self):
        self.triplet_counter = 0

    def convert(self, relationships: List[Dict], nodes: List[Dict]) -> List[Dict]:
        """
        转换关系和节点为三元组格式
        
        Args:
            relationships: WCEP 的关系列表
            nodes: WCEP 的节点列表
        Returns:
            转换后的三元组列表
        """
        # 构建节点 ID 到节点信息的映射
        node_ids = set(node[0] for node in nodes)
        
        triplets = []
        for rel in relationships:
            if len(rel) > 3:
                src = rel[0]
                rel_type = rel[2]
                tgt = rel[3]
            
            if src and rel_type and tgt:
                    triplets.append((src, rel_type, tgt))    
        return triplets
    
    def build_entity_index(self, triplets: List[Dict]) -> Dict[str, List[str]]:
        """
        构建实体索引
        
        Args:
            triplets: 三元组列表
        Returns:
            实体索引字典，键为实体名称，值为包含该实体的三元组列表
        """
        entity_index = defaultdict(list)
        for triplet in triplets:
            src, rel_type, tgt = triplet
            entity_index[src].append(triplet)
            entity_index[tgt].append(triplet)
        return dict(entity_index)
    
    def process_sample(self, sample: Dict) -> Dict:
        """
        处理单个样本，转换关系和节点为三元组格式，并构建实体索引
        
        Args:
            sample: 包含 "relationships" 和 "nodes" 的样本字典
        Returns:
            包含 "triplets" 和 "entity_index" 的处理结果字典
        """
        sample_id = sample.get("sample_id")
        sample_summary = sample.get("sample_summary","")

        all_triplets = []
        articles_with_triplets = []
        for article in sample.get("articles", []):
            graph = article.get("graph", {})
            relationships = graph.get("relationships", [])
            nodes = graph.get("nodes", [])

            triplets = self.convert(relationships, nodes) # 转换为三元组形式
            entity_index = self.build_entity_index(triplets) # 构建文档实体索引

            # 记录文章来源信息
            article_info = {
                "article_id": article.get("article_id"),
                "article_title": article.get("article_title"),
                "triplets": triplets,
                "triplet_count": len(triplets),
                "entity_index": entity_index
            }

            articles_with_triplets.append(article_info)
            all_triplets.extend(triplets)
        
        all_triplets = list(set(all_triplets)) # 去重
        sample_entity_index = self.build_entity_index(all_triplets) # 构建跨文章实体索引

        # 提取所有实体
        all_entities = set()
        for src, rel, tgt in all_triplets:
            all_entities.add(src)
            all_entities.add(tgt)
        
        return {
            "sample_id": sample_id,
            "sample_summary": sample_summary,
            "articles": articles_with_triplets,
            "all_triplets": all_triplets,
            "total_triplet_count": len(all_triplets),
            "sample_entity_index": sample_entity_index,
            "all_entities": sorted(all_entities)
        }
    
    def convert_file(self, input_path: str, output_path: str):
        """
        转换输入文件中的样本，并将结果保存到输出文件
        
        Args:
            input_path: 输入文件路径，包含 WCEP 格式的样本
            output_path: 输出文件路径，保存转换后的结果
        """
        samples = []
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                sample = json.loads(line.strip())
                samples.append(sample)
        
        print(f"加载了{len(samples)} 个样本，开始转换...")
        
        converted_samples = []
        for sample in samples:
            converted_sample = self.process_sample(sample)
            converted_samples.append(converted_sample)
        
        # 写入输出文件
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in converted_samples:
                sample_to_save = sample.copy()
                # 序列化副本中移除各文章级 entity_index
                articles_clean = []
                for art in sample_to_save.get("articles", []):
                    art_copy = art.copy()
                    art_copy.pop("entity_index", None)
                    articles_clean.append(art_copy)
                sample_to_save["articles"] = articles_clean
                f.write(json.dumps(sample_to_save, ensure_ascii=False) + '\n')
        
        # 同时保存实体索引到单独文件
        index_path = output_path.replace('.jsonl', '_index.jsonl')
        all_indices = {}
        for sample in converted_samples:
            sample_id = sample["sample_id"]
            all_indices[sample_id] = {
                "sample_entity_index": sample["sample_entity_index"],
                "all_entities": sample["all_entities"]
            }
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(all_indices, f, ensure_ascii=False, indent=2)
        return converted_samples

if __name__ == "__main__":
    converter = TripletConverter()
    input_file = "..\..\TMP\wcep_kg_4\wcep_graph_results.jsonl" # 输入文件路径
    output_file = "..\..\TMP\wcep_kg_4\step1_output.jsonl" # 输出文件路径
    results = converter.convert_file(input_file, output_file)
    if results:
        print(f"转换完成，结果已保存到 {output_file}")
        print(f"样本示例：{json.dumps(results[0], ensure_ascii=False, indent=2)}")
