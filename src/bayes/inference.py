#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯推断
基于CPD进行概率推断
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.utils.logging import setup_logger

logger = setup_logger("bayes_inference")


class BayesianInference:
    """
    贝叶斯推断器
    
    支持简单的概率查询和后验概率计算
    注意：这里不做分类决策，只输出后验概率
    """
    
    def __init__(self, structure, cpd_learner):
        """
        初始化推断器
        
        Args:
            structure: BayesianNetworkStructure对象
            cpd_learner: CPDLearner对象
        """
        self.structure = structure
        self.cpd_learner = cpd_learner
        logger.info("初始化贝叶斯推断器")
    
    def infer_posterior(
        self, 
        df: pd.DataFrame, 
        target_variable: str = 'weak_label'
    ) -> pd.DataFrame:
        """
        计算目标变量的后验概率
        
        Args:
            df: 包含观测特征的DataFrame
            target_variable: 目标变量名
            
        Returns:
            添加了后验概率列的DataFrame
        """
        logger.info(f"开始计算 {target_variable} 的后验概率...")
        
        df = df.copy()
        
        # 获取目标变量的父节点
        parents = self.structure.get_parents(target_variable)
        
        if target_variable not in self.cpd_learner.cpds:
            logger.warning(f"目标变量 {target_variable} 的CPD未学习")
            df[f'{target_variable}_posterior_prob'] = np.nan
            return df
        
        # 逐行计算后验概率
        posterior_probs = []
        
        for idx, row in df.iterrows():
            # 提取父节点的值,将NaN替换为'MISSING'
            parent_values = {}
            for p in parents:
                if p in row:
                    val = row[p]
                    # 将NaN替换为'MISSING'
                    if pd.isna(val):
                        parent_values[p] = 'MISSING'
                    else:
                        parent_values[p] = val
            
            # 查询 P(target=1 | parents)
            prob_fake = self.cpd_learner.query_cpd(
                target_variable, 
                1,  # 假设1表示虚假
                parent_values
            )
            
            posterior_probs.append(prob_fake)
        
        df[f'{target_variable}_posterior_prob'] = posterior_probs
        
        logger.info("后验概率计算完成")
        return df
    
    def predict(
        self, 
        evidence: Dict[str, any], 
        target_variable: str = 'weak_label'
    ) -> Dict[any, float]:
        """
        单样本预测
        
        Args:
            evidence: 观测证据（特征字典）
            target_variable: 目标变量名
            
        Returns:
            目标变量各状态的概率分布
        """
        parents = self.structure.get_parents(target_variable)
        parent_values = {p: evidence.get(p, None) for p in parents}
        
        cpd = self.cpd_learner.cpds.get(target_variable, None)
        if cpd is None:
            logger.warning(f"目标变量 {target_variable} 的CPD未学习")
            return {}
        
        # 获取目标变量的所有可能状态
        if cpd['type'] == 'prior':
            return cpd['probabilities']
        else:
            parent_key = tuple(parent_values.get(p, None) for p in cpd['parents'])
            return cpd['probabilities'].get(parent_key, {})
    
    def explain_prediction(
        self, 
        evidence: Dict[str, any], 
        target_variable: str = 'weak_label'
    ) -> Dict:
        """
        解释预测结果
        
        返回Markov Blanket和特征贡献
        
        Args:
            evidence: 观测证据
            target_variable: 目标变量
            
        Returns:
            解释字典
        """
        markov_blanket = self.structure.get_markov_blanket(target_variable)
        
        # 预测概率
        prediction = self.predict(evidence, target_variable)
        
        # 提取Markov Blanket中的证据
        mb_evidence = {k: v for k, v in evidence.items() if k in markov_blanket}
        
        return {
            'target_variable': target_variable,
            'prediction': prediction,
            'markov_blanket': markov_blanket,
            'relevant_evidence': mb_evidence,
            'parents': self.structure.get_parents(target_variable)
        }

