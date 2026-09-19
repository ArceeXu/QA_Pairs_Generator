import json
from tqdm import tqdm
import os
from event_relation_modeling.utils_llm import *
import argparse
from collections import defaultdict
from typing import List, Dict, Set, Tuple


def extract_groups_from_single_article(article: dict) -> List[dict]:
    """
    从单篇文章中提取groups
    
    Args:
        article: 包含 "graph" 字段的文章数据，graph包含nodes和relationships
    
    Returns:
        该文章中提取的groups列表
    """
    item = article.get("graph", {})
    if not item or not item.get("nodes") or not item.get("relationships"):
        return []
    
    text = article.get("article_text", "").lower()
    
    # 验证实体是否出现在文本中
    relations_update = []
    try:
        nodes = [node for node in item.get("nodes", []) if node.lower() in text]
    except:
        print(f"Error processing article {article.get('article_id')}")
        return []
    
    # 遍历所有关系三元组，只保留主语和宾语都在节点列表中的
    for (sub, rel, obj) in item.get("relationships", []):
        if (not obj in nodes) or (not sub in nodes):
            continue
        relations_update.append((sub, rel, obj))
    if not relations_update:
        return []
    
    nodes_id = {node: i for i, node in enumerate(nodes)}                 # 为节点分配索引ID
    relations = list(set([rel for (sub, rel, obj) in relations_update])) # 获取所有唯一的关系类型
    relations_id = {relation: i for i, relation in enumerate(relations)} # 为关系类型分配索引ID
    
    # 邻接矩阵 relation_matrix[i][j] 表示从节点j指向节点i的关系类型索引
    relation_matrix = [[None for _ in range(len(nodes))] for _ in range(len(nodes))]
    for (sub, rel, obj) in relations_update:
        if (not obj in nodes) or (not sub in nodes):
            continue
        obj_id = nodes_id[obj]
        rel_id = relations_id[rel]
        sub_id = nodes_id[sub]
        relation_matrix[obj_id][sub_id] = rel_id # 矩阵索引为[目标节点][源节点]
    
    groups = []
    
    # 寻找 A <- B -> C 型（多个宾语共享同一主语）
    for i in range(len(nodes)): # i 为宾语节点索引
        count_relation = {rel: [] for rel in range(len(relations))} # 每个关系类型对应的源节点列表
        for j in range(len(nodes)): # 遍历所有源节点j，统计与宾语i的关系类型
            if relation_matrix[i][j] is None:
                continue
            count_relation[relation_matrix[i][j]].append(j)
        for rel, ents in count_relation.items():
            if len(ents) >= 2:
                groups.append({
                    "subjects": nodes[i],
                    "objects": [nodes[j] for j in ents],
                    "relation": relations[rel],
                    "pattern": "multiple_objects_single_subject"
                })
    
    # 寻找 A -> B <- C 型（多个主语共享同一宾语）
    for i in range(len(nodes)):
        count_relation = {rel: [] for rel in range(len(relations))}
        for j in range(len(nodes)):
            if relation_matrix[j][i] is None:
                continue
            count_relation[relation_matrix[j][i]].append(j)
        for rel, ents in count_relation.items():
            if len(ents) >= 2:
                groups.append({
                    "subjects": [nodes[j] for j in ents],
                    "objects": nodes[i],
                    "relation": relations[rel],
                    "pattern": "multiple_subjects_single_object"
                })
    
    return groups


def normalize_entity_name(entity: str) -> str:
    """实体名称归一化，用于跨文章对齐"""
    normalized = entity.strip().lower()
    # 去除常见标点符号
    import re
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return normalized


def merge_groups_within_sample(groups_with_context: List[dict]) -> List[dict]:
    """
    在同一个样本内，跨多篇文章合并共同性实体组
    
    Args:
        groups_with_context: 列表，每个元素包含 {"article_id": str, "groups": List[dict]}
    
    Returns:
        合并后的跨文章groups列表
    """
    if not groups_with_context:
        return []
    
    # 按关系类型分组
    relation_groups = defaultdict(list)
    for item in groups_with_context:
        article_id = item["article_id"]
        for group in item["groups"]:
            key = group["relation"]
            relation_groups[key].append({
                "article_id": article_id,
                "group": group
            })
    
    merged_groups = []
    
    for relation, group_items in relation_groups.items():
        # 模式1: 多个宾语共享同一主语 (subjects是字符串)
        pattern1_items = [g for g in group_items if isinstance(g["group"]["subjects"], str)]
        
        # 模式2: 多个主语共享同一宾语 (objects是字符串)
        pattern2_items = [g for g in group_items if isinstance(g["group"]["objects"], str)]
        
        # --- 处理模式1: 合并相同主语的groups ---
        subject_to_objects = defaultdict(set)
        subject_to_articles = defaultdict(set)
        subject_raw_map = {}  # 保存原始实体名称
        
        for item in pattern1_items:
            group = item["group"]
            subject = normalize_entity_name(group["subjects"])
            objects = [normalize_entity_name(obj) for obj in group["objects"]]
            
            subject_to_objects[subject].update(objects)
            subject_to_articles[subject].add(item["article_id"])
            if subject not in subject_raw_map:
                subject_raw_map[subject] = group["subjects"]
        
        for subject, objects in subject_to_objects.items():
            if len(objects) >= 2:  # 至少有两个共同宾语
                merged_groups.append({
                    "subjects": subject_raw_map.get(subject, subject),
                    "objects": list(objects),
                    "relation": relation,
                    "source_articles": list(subject_to_articles[subject]),
                    "num_articles": len(subject_to_articles[subject]),
                    "pattern": "multiple_objects_single_subject",
                    "total_objects": len(objects)
                })
        
        # --- 处理模式2: 合并相同宾语的groups ---
        object_to_subjects = defaultdict(set)
        object_to_articles = defaultdict(set)
        object_raw_map = {}
        
        for item in pattern2_items:
            group = item["group"]
            obj = normalize_entity_name(group["objects"])
            subjects = [normalize_entity_name(sub) for sub in group["subjects"]]
            
            object_to_subjects[obj].update(subjects)
            object_to_articles[obj].add(item["article_id"])
            if obj not in object_raw_map:
                object_raw_map[obj] = group["objects"]
        
        for obj, subjects in object_to_subjects.items():
            if len(subjects) >= 2:  # 至少有两个共同主语
                merged_groups.append({
                    "subjects": list(subjects),
                    "objects": object_raw_map.get(obj, obj),
                    "relation": relation,
                    "source_articles": list(object_to_articles[obj]),
                    "num_articles": len(object_to_articles[obj]),
                    "pattern": "multiple_subjects_single_object",
                    "total_subjects": len(subjects)
                })
    
    return merged_groups


def extract_groups_from_sample(sample: dict) -> dict:
    """
    从一个样本（包含多篇文章）中提取共同性实体
    
    Args:
        sample: 包含多篇文章的样本数据（wcep_graph_llm.py的输出格式）
    
    Returns:
        包含该样本所有文章和合并后groups的结果
    """
    sample_id = sample.get("sample_id")
    articles = sample.get("articles", [])
    
    if not articles:
        return {
            "sample_id": sample_id,
            "sample_date": sample.get("sample_date"),
            "sample_summary": sample.get("sample_summary"),
            "sample_category": sample.get("sample_category"),
            "num_articles": 0,
            "articles_groups": [],
            "cross_article_groups": [],
            "total_groups": 0
        }
    
    # 为每篇文章提取groups
    articles_groups = []
    all_groups_with_context = []
    
    for article in articles:
        article_id = article.get("article_id")
        article_groups = extract_groups_from_single_article(article)
        
        articles_groups.append({
            "article_id": article_id,
            "article_title": article.get("article_title"),
            "article_time": article.get("article_time"),
            "article_origin": article.get("article_origin"),
            "num_groups": len(article_groups),
            "groups": article_groups
        })
        
        # 收集用于跨文章合并
        if article_groups:
            all_groups_with_context.append({
                "article_id": article_id,
                "groups": article_groups
            })
    
    # 跨文章合并共同性实体
    cross_article_groups = merge_groups_within_sample(all_groups_with_context)
    
    # 统计信息
    total_groups = sum(len(ag["groups"]) for ag in articles_groups)
    
    return {
        "sample_id": sample_id,
        "sample_date": sample.get("sample_date"),
        "sample_summary": sample.get("sample_summary"),
        "sample_category": sample.get("sample_category"),
        "num_articles": len(articles),
        "num_articles_with_groups": sum(1 for ag in articles_groups if ag["num_groups"] > 0),
        "articles_groups": articles_groups,
        "cross_article_groups": cross_article_groups,
        "total_article_groups": total_groups,
        "total_cross_article_groups": len(cross_article_groups)
    }


def extract_all_samples(src_dir: str, dst_dir: str):
    """
    处理目录下所有WCEP样本文件
    
    Args:
        src_dir: 包含wcep_graph_results.jsonl的目录
        dst_dir: 输出目录
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    # 查找输入文件
    input_file = None
    for f in os.listdir(src_dir):
        if f.endswith(".jsonl") and ("graph" in f or "wcep" in f):
            input_file = f"{src_dir}/{f}"
            break
    
    if not input_file:
        print(f"Error: No JSONL file found in {src_dir}")
        return
    
    print(f"Loading samples from {input_file}...")
    samples = read_jsonl(input_file)
    print(f"Loaded {len(samples)} samples")
    
    all_sample_results = []
    
    for sample in tqdm(samples, desc="Processing samples"):
        result = extract_groups_from_sample(sample)
        all_sample_results.append(result)
    
    # 保存结果
    output_file = f"{dst_dir}/extracted_groups.jsonl"
    write_jsonl(all_sample_results, output_file)
    
    # 保存汇总统计
    summary = {
        "total_samples": len(all_sample_results),
        "total_article_groups": sum(r["total_article_groups"] for r in all_sample_results),
        "total_cross_article_groups": sum(r["total_cross_article_groups"] for r in all_sample_results),
        "samples_with_cross_groups": sum(1 for r in all_sample_results if r["total_cross_article_groups"] > 0),
        "avg_groups_per_sample": sum(r["total_article_groups"] for r in all_sample_results) / len(all_sample_results) if all_sample_results else 0,
        "avg_cross_groups_per_sample": sum(r["total_cross_article_groups"] for r in all_sample_results) / len(all_sample_results) if all_sample_results else 0
    }
    
    summary_file = f"{dst_dir}/summary.json"
    write_json(summary, summary_file, indent=2)
    
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"处理样本数: {summary['total_samples']}")
    print(f"文章内groups总数: {summary['total_article_groups']}")
    print(f"跨文章groups总数: {summary['total_cross_article_groups']}")
    print(f"包含跨文章groups的样本数: {summary['samples_with_cross_groups']}")
    print(f"输出文件: {output_file}")
    print(f"统计文件: {summary_file}")
    
    return all_sample_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True,
                        help="输入目录")
    parser.add_argument("--dst_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--output_format", type=str, default="both",
                        choices=["article_only", "cross_only", "both"],
                        help="输出格式：仅文章内groups / 仅跨文章groups / 两者都输出")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    if args.output_format == "article_only":
        # 仅提取文章内groups，不进行跨文章合并
        samples = read_jsonl(f"{args.src_dir}/wcep_graph_results.jsonl")
        results = []
        for sample in tqdm(samples):
            sample_result = extract_groups_from_sample(sample)
            # 只保留文章内groups
            sample_result["cross_article_groups"] = []
            results.append(sample_result)
        write_jsonl(results, f"{args.dst_dir}/article_groups_only.jsonl")
        
    elif args.output_format == "cross_only":
        # 仅提取跨文章groups
        samples = read_jsonl(f"{args.src_dir}/wcep_graph_results.jsonl")
        results = []
        for sample in tqdm(samples):
            sample_result = extract_groups_from_sample(sample)
            results.append({
                "sample_id": sample_result["sample_id"],
                "sample_summary": sample_result["sample_summary"],
                "cross_article_groups": sample_result["cross_article_groups"]
            })
        write_jsonl(results, f"{args.dst_dir}/cross_article_groups_only.jsonl")
        
    else:  # both
        extract_all_samples(args.src_dir, args.dst_dir)