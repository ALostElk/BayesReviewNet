#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变量离散化模块
将连续变量离散化为有限状态，适用于贝叶斯网络
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json

from src.utils import setup_logger

logger = setup_logger("discretizer")


class FeatureDiscretizer:
    """特征离散化器"""
    
    def __init__(self, discretization_config: Dict[str, Dict[str, Any]]):
        """
        初始化离散化器
        
        Args:
            discretization_config: 离散化配置字典
        """
        self.config = discretization_config
        self.discretization_rules = {}
    
    def discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        离散化DataFrame中的连续特征
        
        Args:
            df: 包含连续特征的DataFrame
            
        Returns:
            添加了离散化特征的DataFrame
        """
        logger.info("开始离散化特征...")
        
        df = df.copy()
        
        # 对每个配置的特征进行离散化
        for feature_name, config in self.config.items():
            if feature_name not in df.columns:
                logger.warning(f"特征 {feature_name} 不存在于DataFrame中，跳过")
                continue
            
            # 执行离散化
            discrete_col_name = f"{feature_name}_discrete"
            df[discrete_col_name] = self._discretize_feature(
                df[feature_name],
                config['bins'],
                config['labels']
            )
            
            # 记录离散化规则
            self.discretization_rules[feature_name] = {
                'bins': config['bins'],
                'labels': config['labels'],
                'discrete_column': discrete_col_name
            }
            
            logger.info(f"特征 {feature_name} 离散化完成 -> {discrete_col_name}")
        
        logger.info("特征离散化完成")
        return df
    
    @staticmethod
    def _discretize_feature(series: pd.Series, bins: List[float], labels: List[str]) -> pd.Series:
        """
        离散化单个特征
        
        Args:
            series: 连续特征序列
            bins: 分箱边界
            labels: 分箱标签
            
        Returns:
            离散化后的序列
        """
        return pd.cut(
            series,
            bins=bins,
            labels=labels,
            include_lowest=True,
            duplicates='drop'
        )
    
    def get_discretization_rules(self) -> Dict[str, Any]:
        """
        获取离散化规则
        
        Returns:
            离散化规则字典
        """
        return self.discretization_rules
    
    def save_rules(self, output_path: str) -> None:
        """
        保存离散化规则到文件
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.discretization_rules, f, indent=2, ensure_ascii=False)
        logger.info(f"离散化规则已保存到 {output_path}")


def create_discretization_summary(df: pd.DataFrame, discretization_rules: Dict) -> Dict[str, Any]:
    """
    创建离散化摘要统计
    
    Args:
        df: 包含离散化特征的DataFrame
        discretization_rules: 离散化规则
        
    Returns:
        摘要统计字典
    """
    summary = {}
    
    for feature_name, rule in discretization_rules.items():
        discrete_col = rule['discrete_column']
        
        if discrete_col not in df.columns:
            continue
        
        # 统计每个离散状态的数量和比例
        value_counts = df[discrete_col].value_counts()
        value_props = df[discrete_col].value_counts(normalize=True)
        
        summary[feature_name] = {
            'original_feature': feature_name,
            'discrete_feature': discrete_col,
            'bins': rule['bins'],
            'labels': rule['labels'],
            'distribution': {
                label: {
                    'count': int(value_counts.get(label, 0)),
                    'proportion': float(value_props.get(label, 0))
                }
                for label in rule['labels']
            },
            'missing_count': int(df[discrete_col].isna().sum()),
            'missing_proportion': float(df[discrete_col].isna().mean())
        }
    
    return summary

