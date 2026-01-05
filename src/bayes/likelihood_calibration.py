#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨域似然校准模块
用于在保持源域先验不变的情况下，使用目标域数据校准特征的条件分布
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from src.utils.logging import setup_logger

logger = setup_logger("likelihood_calibrator")


class LikelihoodCalibrator:
    """
    似然校准器
    
    核心思想：
    1. 保持源域学到的DAG结构和P(Fraud)先验不变
    2. 使用少量目标域数据重新估计P(features | Fraud)
    3. 不改变判别边界，只调整特征分布参数
    """
    
    def __init__(self, source_cpd_learner, calibration_strength: float = 0.3):
        """
        初始化校准器
        
        Args:
            source_cpd_learner: 源域训练的CPD学习器（Amazon）
            calibration_strength: 校准强度 α ∈ [0,1]
                                 α=0: 完全保持源域分布
                                 α=1: 完全使用目标域分布
                                 α=0.3: 轻度校准（推荐）
        """
        self.source_cpd = source_cpd_learner
        self.calibration_strength = calibration_strength
        self.calibrated_cpds = {}
        
        logger.info("初始化似然校准器")
        logger.info(f"  校准强度 α = {calibration_strength}")
    
    def calibrate(
        self,
        calibration_df: pd.DataFrame,
        target_variable: str = 'weak_label',
        feature_nodes: Optional[List[str]] = None
    ):
        """
        执行似然校准
        
        Args:
            calibration_df: 目标域校准数据（小规模，5%~15%）
            target_variable: 目标变量（Fraud标签）
            feature_nodes: 需要校准的特征节点列表
                          None表示校准所有非目标节点
        """
        logger.info("=" * 80)
        logger.info("开始跨域似然校准")
        logger.info("=" * 80)
        
        # 确定需要校准的节点
        if feature_nodes is None:
            # 校准所有特征节点（非目标节点）
            feature_nodes = [
                node for node in self.source_cpd.cpds.keys()
                if node != target_variable
            ]
        
        logger.info(f"校准数据集大小: {len(calibration_df)} 条")
        logger.info(f"需要校准的节点: {len(feature_nodes)} 个")
        
        # 复制源域CPD作为基础
        self.calibrated_cpds = self.source_cpd.cpds.copy()
        
        # 对每个特征节点进行校准
        n_calibrated = 0
        for node in feature_nodes:
            if node not in self.source_cpd.cpds:
                logger.warning(f"节点 {node} 不在源域CPD中，跳过")
                continue
            
            success = self._calibrate_node(node, calibration_df, target_variable)
            if success:
                n_calibrated += 1
        
        logger.info(f"✓ 校准完成: {n_calibrated}/{len(feature_nodes)} 个节点")
        logger.info("=" * 80)
    
    def _calibrate_node(
        self,
        node: str,
        calibration_df: pd.DataFrame,
        target_variable: str
    ) -> bool:
        """
        校准单个节点的CPD
        
        核心公式：
        P_calibrated(feature | Fraud) = 
            (1-α) * P_source(feature | Fraud) + 
            α * P_target(feature | Fraud)
        
        Args:
            node: 节点名
            calibration_df: 校准数据
            target_variable: 目标变量
            
        Returns:
            是否成功校准
        """
        if node not in calibration_df.columns:
            logger.debug(f"节点 {node} 不在校准数据中，保持源域分布")
            return False
        
        source_cpd = self.source_cpd.cpds[node]
        
        # 获取节点的父节点
        parents = source_cpd.get('parents', [])
        
        # 如果父节点中包含目标变量，需要按条件校准
        if target_variable in parents:
            return self._calibrate_conditional_on_target(
                node, calibration_df, target_variable, parents
            )
        else:
            # 否则，校准边际分布或其他条件分布
            return self._calibrate_marginal_or_other(
                node, calibration_df, parents
            )
    
    def _calibrate_conditional_on_target(
        self,
        node: str,
        calibration_df: pd.DataFrame,
        target_variable: str,
        parents: List[str]
    ) -> bool:
        """
        校准以目标变量为条件的CPD: P(feature | Fraud, other_parents)
        
        这是核心校准逻辑
        """
        source_cpd = self.source_cpd.cpds[node].copy()
        
        # 从目标域数据估计新的条件分布
        target_cpd = self._estimate_cpd_from_data(
            node, calibration_df, parents
        )
        
        if target_cpd is None or len(target_cpd.get('probabilities', {})) == 0:
            logger.debug(f"节点 {node} 在目标域中无足够数据，保持源域分布")
            return False
        
        # 混合源域和目标域的CPD
        calibrated_probs = self._blend_cpds(
            source_cpd['probabilities'],
            target_cpd['probabilities'],
            alpha=self.calibration_strength
        )
        
        # 更新校准后的CPD
        source_cpd['probabilities'] = calibrated_probs
        self.calibrated_cpds[node] = source_cpd
        
        logger.debug(f"✓ 节点 {node} 校准完成")
        return True
    
    def _calibrate_marginal_or_other(
        self,
        node: str,
        calibration_df: pd.DataFrame,
        parents: List[str]
    ) -> bool:
        """
        校准边际分布或不依赖目标变量的条件分布
        
        虽然这些节点不直接依赖Fraud，但它们的分布会间接影响推断
        因此也需要校准
        """
        source_cpd = self.source_cpd.cpds[node].copy()
        
        # 从目标域数据估计新的条件分布
        target_cpd = self._estimate_cpd_from_data(
            node, calibration_df, parents
        )
        
        if target_cpd is None or len(target_cpd.get('probabilities', {})) == 0:
            logger.debug(f"节点 {node} 在目标域中无足够数据，保持源域分布")
            return False
        
        # 混合源域和目标域的CPD
        calibrated_probs = self._blend_cpds(
            source_cpd['probabilities'],
            target_cpd['probabilities'],
            alpha=self.calibration_strength
        )
        
        # 更新校准后的CPD
        source_cpd['probabilities'] = calibrated_probs
        self.calibrated_cpds[node] = source_cpd
        
        logger.debug(f"✓ 节点 {node} 校准完成")
        return True
    
    def _estimate_cpd_from_data(
        self,
        node: str,
        df: pd.DataFrame,
        parents: List[str],
        smoothing: float = 1.0
    ) -> Optional[Dict]:
        """
        从数据中估计CPD（与CPDLearner类似，但独立实现）
        """
        # 过滤有效数据
        valid_cols = [node] + [p for p in parents if p in df.columns]
        df_valid = df[valid_cols].dropna()
        
        # 填充父节点的NaN为'MISSING'
        for parent in parents:
            if parent in df_valid.columns:
                if pd.api.types.is_categorical_dtype(df_valid[parent]):
                    if 'MISSING' not in df_valid[parent].cat.categories:
                        df_valid[parent] = df_valid[parent].cat.add_categories(['MISSING'])
                df_valid[parent] = df_valid[parent].fillna('MISSING')
        
        if len(df_valid) == 0:
            return None
        
        # 获取节点状态
        node_states = df_valid[node].unique().tolist()
        
        # 按父节点分组统计
        cpd_table = defaultdict(dict)
        parent_cols = [p for p in parents if p in df_valid.columns]
        
        if len(parent_cols) == 0:
            # 无父节点，先验分布
            value_counts = df_valid[node].value_counts()
            n = len(df_valid)
            k = len(node_states)
            
            for state in node_states:
                count = value_counts.get(state, 0)
                prob = (count + smoothing) / (n + smoothing * k)
                cpd_table[()][state] = prob
        else:
            # 有父节点，条件分布
            for parent_values, group in df_valid.groupby(parent_cols):
                if not isinstance(parent_values, tuple):
                    parent_values = (parent_values,)
                
                value_counts = group[node].value_counts()
                n = len(group)
                k = len(node_states)
                
                for state in node_states:
                    count = value_counts.get(state, 0)
                    prob = (count + smoothing) / (n + smoothing * k)
                    cpd_table[parent_values][state] = prob
        
        return {
            'type': 'prior' if len(parents) == 0 else 'conditional',
            'node': node,
            'parents': parents,
            'node_states': node_states,
            'probabilities': dict(cpd_table)
        }
    
    def _blend_cpds(
        self,
        source_probs: Dict,
        target_probs: Dict,
        alpha: float
    ) -> Dict:
        """
        混合源域和目标域的概率分布
        
        P_blended = (1-α) * P_source + α * P_target
        
        Args:
            source_probs: 源域概率表
            target_probs: 目标域概率表
            alpha: 混合系数
            
        Returns:
            混合后的概率表
        """
        blended_probs = {}
        
        # 遍历所有父节点组合
        all_keys = set(source_probs.keys()) | set(target_probs.keys())
        
        for parent_key in all_keys:
            source_dist = source_probs.get(parent_key, {})
            target_dist = target_probs.get(parent_key, {})
            
            # 如果目标域中没有这个组合，完全使用源域
            if not target_dist:
                blended_probs[parent_key] = source_dist
                continue
            
            # 混合两个分布
            all_states = set(source_dist.keys()) | set(target_dist.keys())
            blended_dist = {}
            
            for state in all_states:
                p_source = source_dist.get(state, 0.0)
                p_target = target_dist.get(state, 0.0)
                
                # 线性插值
                p_blended = (1 - alpha) * p_source + alpha * p_target
                blended_dist[state] = p_blended
            
            # 归一化（确保概率和为1）
            total = sum(blended_dist.values())
            if total > 0:
                blended_dist = {k: v/total for k, v in blended_dist.items()}
            
            blended_probs[parent_key] = blended_dist
        
        return blended_probs
    
    def get_calibrated_cpd_learner(self):
        """
        返回一个包含校准后CPD的CPD学习器对象
        
        这个对象可以直接用于后续的推断
        """
        # 创建一个新的CPD学习器对象
        from src.bayes.cpds import CPDLearner
        
        calibrated_learner = CPDLearner(self.source_cpd.structure)
        calibrated_learner.cpds = self.calibrated_cpds.copy()
        
        return calibrated_learner
    
    def get_calibration_report(self) -> Dict:
        """
        生成校准报告
        
        Returns:
            包含校准统计信息的字典
        """
        n_total = len(self.source_cpd.cpds)
        n_calibrated = len([
            node for node in self.calibrated_cpds
            if node in self.source_cpd.cpds and 
            self.calibrated_cpds[node] != self.source_cpd.cpds[node]
        ])
        
        return {
            'total_nodes': n_total,
            'calibrated_nodes': n_calibrated,
            'calibration_strength': self.calibration_strength,
            'kept_nodes': n_total - n_calibrated
        }

