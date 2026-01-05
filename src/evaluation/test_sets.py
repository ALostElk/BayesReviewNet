#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试集处理
支持Yelp等数据集的加载与统计
"""
import pandas as pd
from typing import Dict

from src.utils.logging import setup_logger

logger = setup_logger("test_sets")


class YelpTestSet:
    """
    Yelp测试集
    
    支持多视角特征：Text + Behavior + Network
    使用平台标签或启发式规则作为弱监督信号
    """
    
    def __init__(self, data_path: str):
        """
        初始化测试集
        
        Args:
            data_path: 数据路径
        """
        self.data_path = data_path
        self.df = None
        logger.info(f"初始化Yelp测试集: {data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        加载测试集
        
        Returns:
            测试集DataFrame
        """
        logger.info("加载Yelp测试集...")
        
        if self.data_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path}")
        
        logger.info(f"Yelp测试集加载完成，共 {len(self.df)} 条记录")
        return self.df
    
    def get_weak_labels(self) -> pd.Series:
        """
        获取弱标签
        
        Returns:
            标签Series
        """
        if self.df is None:
            self.load()
        
        return self.df['weak_label']
    
    def get_statistics(self) -> Dict:
        """
        获取测试集统计信息
        
        Returns:
            统计信息字典
        """
        if self.df is None:
            self.load()
        
        stats = {
            'total_samples': len(self.df),
            'suspicious_samples': int(self.df['weak_label'].sum()) if 'weak_label' in self.df.columns else 0,
            'suspicious_ratio': float(self.df['weak_label'].mean()) if 'weak_label' in self.df.columns else 0.0,
            'label_source': self.df['label_source'].value_counts().to_dict() if 'label_source' in self.df.columns else {}
        }
        
        # 添加特征覆盖率统计
        feature_types = {
            'text_features': ['review_length', 'sentiment_score', 'subjectivity_score'],
            'behavior_features': ['user_review_count', 'user_rating_deviation', 'user_rating_entropy'],
            'network_features': []  # 未来扩展
        }
        
        for feat_type, features in feature_types.items():
            available = sum(1 for f in features if f in self.df.columns)
            stats[f'{feat_type}_coverage'] = f"{available}/{len(features)}" if features else "0/0"
        
        return stats
