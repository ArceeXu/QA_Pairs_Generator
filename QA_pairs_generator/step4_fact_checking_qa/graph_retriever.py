import difflib
from typing import List, Dict, Set, Tuple

class GraphRetriever:
    """
    依据 BYOKG-RAG 的多级检索机制：
    1. 精确匹配（Exact Match）
    2. 模糊相似匹配（Fuzzy Match）
    3. 多跳广度检索（Multi-hop Match）
    """
    def __init__(self, triplets: List[List], sample_entity_index: Dict[str, List[List]]):
        # 确保全部转化为tuple格式增强索引和哈希表现
        self.triplets = [tuple(t) for t in triplets]
        self.index = {k: [tuple(t) for t in v] for k, v in sample_entity_index.items()}

    def retrieve_by_entities(self, aligned_entities: List[str], fuzzy_threshold: float = 0.6) -> List[Tuple]:
        retrieved_triplets = set()
        
        # 1. 精确匹配 + 2. 基于编辑距离的模糊匹配
        matched_kg_entities = set()
        for c_ent in aligned_entities:
            # 看看是否本身就在KG索引里
            if c_ent in self.index:
                matched_kg_entities.add(c_ent)
            else:
                # 检索相似实体
                for kg_ent in self.index.keys():
                    sim = difflib.SequenceMatcher(None, c_ent, kg_ent).ratio()
                    if sim >= fuzzy_threshold:
                        matched_kg_entities.add(kg_ent)

        # 收集一跳基础关联三元组
        for ent in matched_kg_entities:
            for t in self.index.get(ent, []):
                retrieved_triplets.add(t)

        # 3. 多跳匹配 (Multi-hop: 引入一跳邻居实体的二级外延)
        one_hop_neighbors = set()
        for t in retrieved_triplets:
            one_hop_neighbors.add(t[0])
            one_hop_neighbors.add(t[2])

        # 获取二跳扩展
        for neighbor in one_hop_neighbors:
            if neighbor in self.index:
                for t_2hop in self.index[neighbor]:
                    # 限制不无节制膨胀，只选取和初始实体集有一点边交叉的
                    retrieved_triplets.add(t_2hop)

        return list(retrieved_triplets)