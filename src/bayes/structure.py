#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯网络DAG结构定义
定义节点间的因果关系
"""
from typing import List, Tuple, Dict
import networkx as nx

from src.utils.logging import setup_logger

logger = setup_logger("bayes_structure")


class BayesianNetworkStructure:
    """
    贝叶斯网络DAG结构
    
    定义变量之间的因果关系（父子节点）
    """
    
    def __init__(self):
        """初始化网络结构"""
        self.graph = nx.DiGraph()
        self.edges = []
        logger.info("初始化贝叶斯网络结构")
    
    def define_structure(self, structure_type: str = 'default') -> None:
        """
        定义网络结构
        
        Args:
            structure_type: 结构类型
                - 'default': 默认因果结构
                - 'naive': 朴素贝叶斯结构
                - 'custom': 自定义结构
        """
        if structure_type == 'default':
            self._define_default_structure()
        elif structure_type == 'naive':
            self._define_naive_bayes_structure()
        else:
            raise ValueError(f"未知的结构类型: {structure_type}")
        
        logger.info(f"网络结构已定义: {structure_type}, 共 {len(self.edges)} 条边")
    
    def _define_default_structure(self) -> None:
        """
        定义默认因果结构
        
        因果假设:
        1. 用户行为 -> 弱标签
        2. 文本特征 -> 弱标签
        3. 用户行为 -> 文本特征（间接影响）
        4. 平台特定 -> 弱标签
        """
        # 用户行为 -> 弱标签
        self.add_edge('user_review_count_discrete', 'weak_label')
        self.add_edge('user_rating_deviation_discrete', 'weak_label')
        self.add_edge('user_rating_entropy_discrete', 'weak_label')
        self.add_edge('user_review_burstiness_discrete', 'weak_label')
        
        # 文本特征 -> 弱标签
        self.add_edge('review_length_discrete', 'weak_label')
        self.add_edge('sentiment_score_discrete', 'weak_label')
        self.add_edge('subjectivity_score_discrete', 'weak_label')
        
        # 用户行为 -> 文本特征（虚假评论者可能有固定的文本模式）
        self.add_edge('user_review_count_discrete', 'review_length_discrete')
        self.add_edge('user_rating_deviation_discrete', 'sentiment_score_discrete')
        
        # 平台特定 -> 弱标签（Amazon）
        self.add_edge('verified', 'weak_label')
    
    def _define_naive_bayes_structure(self) -> None:
        """
        定义朴素贝叶斯结构
        
        所有特征都直接指向标签，特征间相互独立
        """
        features = [
            'user_review_count_discrete',
            'user_rating_deviation_discrete',
            'user_rating_entropy_discrete',
            'user_review_burstiness_discrete',
            'review_length_discrete',
            'sentiment_score_discrete',
            'subjectivity_score_discrete',
            'verified'
        ]
        
        for feature in features:
            self.add_edge(feature, 'weak_label')
    
    def add_edge(self, parent: str, child: str) -> None:
        """
        添加有向边（因果关系）
        
        Args:
            parent: 父节点（原因）
            child: 子节点（结果）
        """
        self.graph.add_edge(parent, child)
        self.edges.append((parent, child))
        logger.debug(f"添加边: {parent} -> {child}")
    
    def get_parents(self, node: str) -> List[str]:
        """
        获取节点的父节点
        
        Args:
            node: 节点名
            
        Returns:
            父节点列表
        """
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """
        获取节点的子节点
        
        Args:
            node: 节点名
            
        Returns:
            子节点列表
        """
        return list(self.graph.successors(node))
    
    def get_markov_blanket(self, node: str) -> List[str]:
        """
        获取Markov Blanket
        
        包含：父节点、子节点、子节点的其他父节点
        
        Args:
            node: 节点名
            
        Returns:
            Markov Blanket节点列表
        """
        markov_blanket = set()
        
        # 父节点
        markov_blanket.update(self.get_parents(node))
        
        # 子节点
        children = self.get_children(node)
        markov_blanket.update(children)
        
        # 子节点的其他父节点
        for child in children:
            markov_blanket.update(self.get_parents(child))
        
        # 移除节点自身
        markov_blanket.discard(node)
        
        return list(markov_blanket)
    
    def is_acyclic(self) -> bool:
        """检查是否为有向无环图"""
        return nx.is_directed_acyclic_graph(self.graph)
    
    def get_topological_order(self) -> List[str]:
        """获取拓扑排序"""
        if not self.is_acyclic():
            raise ValueError("图中存在环，无法进行拓扑排序")
        return list(nx.topological_sort(self.graph))
    
    def export_structure(self) -> Dict:
        """
        导出网络结构
        
        Returns:
            结构字典
        """
        return {
            'nodes': list(self.graph.nodes()),
            'edges': self.edges,
            'is_acyclic': self.is_acyclic(),
            'topological_order': self.get_topological_order() if self.is_acyclic() else None
        }

