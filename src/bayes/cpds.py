#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
条件概率分布（CPD）学习
从数据中学习CPD表
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from src.utils.logging import setup_logger

logger = setup_logger("cpd_learner")


class CPDLearner:
    """
    条件概率分布学习器
    
    从离散化数据中估计条件概率表
    """
    
    def __init__(self, structure):
        """
        初始化CPD学习器
        
        Args:
            structure: BayesianNetworkStructure对象
        """
        self.structure = structure
        self.cpds = {}
        logger.info("初始化CPD学习器")
    
    def learn_cpds(self, df: pd.DataFrame, smoothing: float = 1.0) -> Dict:
        """
        从数据中学习所有CPD
        
        Args:
            df: 包含离散化特征的DataFrame
            smoothing: Laplace平滑参数（防止零概率）
            
        Returns:
            CPD字典
        """
        logger.info("开始学习条件概率分布...")
        
        nodes = list(self.structure.graph.nodes())
        
        for node in nodes:
            if node not in df.columns:
                logger.warning(f"节点 {node} 不在数据中，跳过")
                continue
            
            parents = self.structure.get_parents(node)
            cpd = self._learn_single_cpd(df, node, parents, smoothing)
            self.cpds[node] = cpd
            
            logger.debug(f"节点 {node} 的CPD已学习，父节点: {parents}")
        
        logger.info(f"CPD学习完成，共学习 {len(self.cpds)} 个节点的CPD")
        return self.cpds
    
    def _learn_single_cpd(
        self, 
        df: pd.DataFrame, 
        node: str, 
        parents: List[str], 
        smoothing: float
    ) -> Dict:
        """
        学习单个节点的CPD
        
        Args:
            df: 数据DataFrame
            node: 目标节点
            parents: 父节点列表
            smoothing: 平滑参数
            
        Returns:
            CPD字典
        """
        if not parents:
            # 无父节点，计算先验概率
            return self._learn_prior(df, node, smoothing)
        else:
            # 有父节点，计算条件概率
            return self._learn_conditional(df, node, parents, smoothing)
    
    def _learn_prior(self, df: pd.DataFrame, node: str, smoothing: float) -> Dict:
        """
        学习先验概率 P(node)
        
        Args:
            df: 数据DataFrame
            node: 节点名
            smoothing: 平滑参数
            
        Returns:
            先验概率字典
        """
        # 统计各状态的频数
        value_counts = df[node].value_counts()
        states = value_counts.index.tolist()
        
        # Laplace平滑
        n = len(df)
        k = len(states)
        
        prior = {}
        for state in states:
            count = value_counts.get(state, 0)
            prior[state] = (count + smoothing) / (n + smoothing * k)
        
        return {
            'type': 'prior',
            'node': node,
            'states': states,
            'probabilities': prior
        }
    
    def _learn_conditional(
        self, 
        df: pd.DataFrame, 
        node: str, 
        parents: List[str], 
        smoothing: float
    ) -> Dict:
        """
        学习条件概率 P(node | parents)
        
        Args:
            df: 数据DataFrame
            node: 节点名
            parents: 父节点列表
            smoothing: 平滑参数
            
        Returns:
            条件概率字典
        """
        # 只过滤目标节点的NaN,父节点的NaN用'MISSING'填充
        df_work = df.copy()
        
        # 检查目标节点是否有效
        if node not in df_work.columns or df_work[node].isna().all():
            logger.warning(f"节点 {node} 无有效数据")
            return {
                'type': 'conditional',
                'node': node,
                'parents': parents,
                'probabilities': {}
            }
        
        # 只过滤目标节点的NaN
        df_valid = df_work[df_work[node].notna()].copy()
        
        if len(df_valid) == 0:
            logger.warning(f"节点 {node} 过滤后无有效数据")
            return {
                'type': 'conditional',
                'node': node,
                'parents': parents,
                'probabilities': {}
            }
        
        # 对于父节点,将NaN填充为'MISSING'
        for parent in parents:
            if parent in df_valid.columns:
                # 如果是Categorical类型,需要先添加'MISSING'类别
                if pd.api.types.is_categorical_dtype(df_valid[parent]):
                    if 'MISSING' not in df_valid[parent].cat.categories:
                        df_valid[parent] = df_valid[parent].cat.add_categories(['MISSING'])
                df_valid[parent] = df_valid[parent].fillna('MISSING')
        
        # 获取各变量的状态
        node_states = df_valid[node].unique().tolist()
        
        # 按父节点分组统计
        cpd_table = defaultdict(dict)
        
        # 对每个父节点组合
        parent_cols = [p for p in parents if p in df_valid.columns]
        if len(parent_cols) == 0:
            # 没有父节点,学习边际概率
            value_counts = df_valid[node].value_counts()
            n = len(df_valid)
            k = len(node_states)
            for state in node_states:
                count = value_counts.get(state, 0)
                prob = (count + smoothing) / (n + smoothing * k)
                cpd_table[()][state] = prob
        else:
            # 有父节点,按父节点组合分组
            for parent_values, group in df_valid.groupby(parent_cols):
                if not isinstance(parent_values, tuple):
                    parent_values = (parent_values,)
                
                # 统计子节点各状态的频数
                value_counts = group[node].value_counts()
                n = len(group)
                k = len(node_states)
                
                # 计算条件概率（带平滑）
                for state in node_states:
                    count = value_counts.get(state, 0)
                    prob = (count + smoothing) / (n + smoothing * k)
                    cpd_table[parent_values][state] = prob
        
        return {
            'type': 'conditional',
            'node': node,
            'parents': parents,
            'node_states': node_states,
            'probabilities': dict(cpd_table)
        }
    
    def query_cpd(self, node: str, node_value, parent_values: Optional[Dict] = None) -> float:
        """
        查询CPD
        
        Args:
            node: 节点名
            node_value: 节点取值
            parent_values: 父节点取值字典
            
        Returns:
            概率值
        """
        if node not in self.cpds:
            logger.warning(f"节点 {node} 的CPD未学习")
            return 0.0
        
        cpd = self.cpds[node]
        
        if cpd['type'] == 'prior':
            return cpd['probabilities'].get(node_value, 0.0)
        else:
            # 条件概率
            if parent_values is None:
                return 0.0
            
            # 构造父节点值的tuple key
            parent_key = tuple(parent_values.get(p, None) for p in cpd['parents'])
            
            prob_table = cpd['probabilities'].get(parent_key, {})
            return prob_table.get(node_value, 0.0)
    
    def export_cpds(self) -> Dict:
        """导出所有CPD"""
        return self.cpds

