#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用户行为特征聚合模块
计算用户级别的统计特征并回填到评论表
"""
import pandas as pd
import numpy as np
from typing import Dict
from scipy.stats import entropy
from tqdm import tqdm

from src.utils import setup_logger

logger = setup_logger("user_features")


class UserFeatureAggregator:
    """用户特征聚合器"""
    
    def __init__(self):
        self.feature_names = [
            'user_review_count',
            'user_avg_rating',
            'user_rating_std',
            'user_rating_deviation',
            'user_rating_entropy',
            'user_verified_ratio',
            'user_review_burstiness',
            'user_avg_review_length'
        ]
    
    def aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合用户级特征并回填到评论表
        
        Args:
            df: 评论DataFrame
            
        Returns:
            添加了用户特征的DataFrame
        """
        logger.info("开始聚合用户特征...")
        
        df = df.copy()
        
        # 按用户分组计算特征
        user_features = self._compute_user_features(df)
        
        # 回填到评论表
        for feature_name in self.feature_names:
            if feature_name in user_features.columns:
                df[feature_name] = df['user_id'].map(user_features[feature_name])
        
        logger.info("用户特征聚合完成")
        return df
    
    def _compute_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算用户级特征
        
        Args:
            df: 评论DataFrame
            
        Returns:
            用户特征DataFrame (索引为user_id)
        """
        user_features = pd.DataFrame()
        
        # 评论数
        user_features['user_review_count'] = df.groupby('user_id').size()
        
        # 评分统计
        rating_stats = df.groupby('user_id')['rating'].agg([
            ('user_avg_rating', 'mean'),
            ('user_rating_std', 'std')
        ])
        user_features = user_features.join(rating_stats)
        
        # 评分偏差（相对于全局平均）
        global_avg_rating = df['rating'].mean()
        user_features['user_rating_deviation'] = (
            user_features['user_avg_rating'] - global_avg_rating
        ).abs()
        
        # 评分熵
        user_features['user_rating_entropy'] = df.groupby('user_id')['rating'].apply(
            self._calculate_rating_entropy
        )
        
        # verified比例（仅Amazon）
        if 'verified' in df.columns:
            user_features['user_verified_ratio'] = df.groupby('user_id')['verified'].apply(
                lambda x: x.sum() / len(x) if len(x) > 0 else np.nan
            )
        else:
            user_features['user_verified_ratio'] = np.nan
        
        # 评论爆发性（时间间隔统计）
        user_features['user_review_burstiness'] = df.groupby('user_id').apply(
            self._calculate_burstiness
        )
        
        # 平均评论长度
        if 'review_length' in df.columns:
            user_features['user_avg_review_length'] = df.groupby('user_id')['review_length'].mean()
        else:
            user_features['user_avg_review_length'] = np.nan
        
        return user_features
    
    @staticmethod
    def _calculate_rating_entropy(ratings: pd.Series) -> float:
        """
        计算评分熵
        
        Args:
            ratings: 评分序列
            
        Returns:
            熵值
        """
        if len(ratings) == 0:
            return np.nan
        
        # 计算评分分布
        value_counts = ratings.value_counts()
        probabilities = value_counts / len(ratings)
        
        # 计算熵
        return entropy(probabilities)
    
    @staticmethod
    def _calculate_burstiness(group: pd.DataFrame) -> float:
        """
        计算评论爆发性（平均时间间隔，单位：天）
        
        Args:
            group: 用户的所有评论
            
        Returns:
            平均时间间隔（天）
        """
        if 'timestamp' not in group.columns or len(group) < 2:
            return np.nan
        
        timestamps = group['timestamp'].dropna().sort_values()
        
        if len(timestamps) < 2:
            return np.nan
        
        # 计算相邻评论的时间间隔
        intervals = timestamps.diff().dt.total_seconds() / 86400  # 转换为天
        
        # 返回平均间隔
        return intervals.mean()
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        获取特征描述
        
        Returns:
            特征名称到描述的映射
        """
        return {
            'user_review_count': '用户评论总数',
            'user_avg_rating': '用户平均评分',
            'user_rating_std': '用户评分标准差',
            'user_rating_deviation': '用户评分偏差（相对全局均值）',
            'user_rating_entropy': '用户评分熵（多样性）',
            'user_verified_ratio': '用户验证评论比例',
            'user_review_burstiness': '用户评论爆发性（平均间隔天数）',
            'user_avg_review_length': '用户平均评论长度'
        }

