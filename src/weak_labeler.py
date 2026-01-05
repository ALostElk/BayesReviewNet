#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
弱标签构造模块
基于启发式规则构造弱监督标签
"""
import pandas as pd
import numpy as np
from typing import Dict, List

from src.utils import setup_logger

logger = setup_logger("weak_labeler")


class WeakLabeler:
    """弱标签构造器"""
    
    def __init__(self, platform: str):
        """
        初始化弱标签构造器
        
        Args:
            platform: 平台名称 ('amazon', 'yelp', 'opspam')
        """
        self.platform = platform
    
    def construct_weak_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构造弱标签
        
        Args:
            df: 包含特征的DataFrame
            
        Returns:
            添加了弱标签的DataFrame
        """
        logger.info(f"开始为 {self.platform} 构造弱标签...")
        
        df = df.copy()
        
        if self.platform == 'amazon':
            df = self._construct_amazon_weak_labels(df)
        elif self.platform == 'yelp':
            df = self._construct_yelp_weak_labels(df)
        elif self.platform == 'opspam':
            # OpSpam已有真实标签，不需要构造弱标签
            logger.info("OpSpam数据集已有真实标签，跳过弱标签构造")
        else:
            logger.warning(f"未知平台: {self.platform}")
        
        logger.info("弱标签构造完成")
        return df
    
    def _construct_amazon_weak_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为Amazon数据构造弱标签
        
        基于以下启发式规则：
        1. 未验证购买 (verified=False)
        2. 评分极端 (1星或5星)
        3. 评论过短或过长
        4. 用户评论爆发性高（短时间内大量评论）
        5. 评分与用户历史偏差大
        
        Args:
            df: Amazon评论DataFrame
            
        Returns:
            添加了弱标签的DataFrame
        """
        # 初始化可疑分数
        df['suspicion_score'] = 0
        
        # 规则1: 未验证购买 (+2分)
        if 'verified' in df.columns:
            df.loc[df['verified'] == False, 'suspicion_score'] += 2
        
        # 规则2: 评分极端 (+1分)
        if 'rating' in df.columns:
            df.loc[(df['rating'] == 1) | (df['rating'] == 5), 'suspicion_score'] += 1
        
        # 规则3: 评论长度异常 (+1分)
        if 'review_length' in df.columns:
            # 过短（<30字符）或过长（>2000字符）
            df.loc[(df['review_length'] < 30) | (df['review_length'] > 2000), 'suspicion_score'] += 1
        
        # 规则4: 评论爆发性高 (+2分)
        if 'user_review_burstiness' in df.columns:
            # 平均间隔小于1天
            df.loc[df['user_review_burstiness'] < 1, 'suspicion_score'] += 2
        
        # 规则5: 评分偏差大 (+1分)
        if 'user_rating_deviation' in df.columns:
            # 偏差大于1.5
            df.loc[df['user_rating_deviation'] > 1.5, 'suspicion_score'] += 1
        
        # 规则6: 用户评论数过多 (+1分)
        if 'user_review_count' in df.columns:
            # 评论数超过100
            df.loc[df['user_review_count'] > 100, 'suspicion_score'] += 1
        
        # 根据可疑分数构造弱标签
        # 阈值：>=4分认为可疑
        df['weak_label'] = (df['suspicion_score'] >= 4).astype(int)
        df['label_source'] = 'heuristic_rules'
        
        # 统计
        suspicious_count = df['weak_label'].sum()
        total_count = len(df)
        logger.info(f"Amazon弱标签统计: {suspicious_count}/{total_count} ({suspicious_count/total_count*100:.2f}%) 标记为可疑")
        
        return df
    
    def _construct_yelp_weak_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为Yelp数据构造弱标签
        
        基于以下启发式规则：
        1. 评分极端
        2. 评论长度异常
        3. 评论爆发性高
        4. 评分熵低（用户总是给相同评分）
        5. 第一人称代词使用异常
        
        Args:
            df: Yelp评论DataFrame
            
        Returns:
            添加了弱标签的DataFrame
        """
        # 初始化可疑分数
        df['suspicion_score'] = 0
        
        # 规则1: 评分极端 (+1分)
        if 'rating' in df.columns:
            df.loc[(df['rating'] == 1) | (df['rating'] == 5), 'suspicion_score'] += 1
        
        # 规则2: 评论长度异常 (+1分)
        if 'review_length' in df.columns:
            df.loc[(df['review_length'] < 30) | (df['review_length'] > 2000), 'suspicion_score'] += 1
        
        # 规则3: 评论爆发性高 (+2分)
        if 'user_review_burstiness' in df.columns:
            df.loc[df['user_review_burstiness'] < 1, 'suspicion_score'] += 2
        
        # 规则4: 评分熵低 (+2分)
        if 'user_rating_entropy' in df.columns:
            # 熵小于0.5表示评分模式单一
            df.loc[df['user_rating_entropy'] < 0.5, 'suspicion_score'] += 2
        
        # 规则5: 第一人称代词使用异常 (+1分)
        if 'first_person_pronoun_ratio' in df.columns:
            # 过高（>0.15）或过低（<0.02）
            df.loc[(df['first_person_pronoun_ratio'] > 0.15) | 
                   (df['first_person_pronoun_ratio'] < 0.02), 'suspicion_score'] += 1
        
        # 规则6: 情感极端 (+1分)
        if 'sentiment_score' in df.columns:
            # 情感分数绝对值大于0.8
            df.loc[df['sentiment_score'].abs() > 0.8, 'suspicion_score'] += 1
        
        # 根据可疑分数构造弱标签
        # 阈值：>=4分认为可疑
        df['weak_label'] = (df['suspicion_score'] >= 4).astype(int)
        df['label_source'] = 'heuristic_rules'
        
        # 统计
        suspicious_count = df['weak_label'].sum()
        total_count = len(df)
        logger.info(f"Yelp弱标签统计: {suspicious_count}/{total_count} ({suspicious_count/total_count*100:.2f}%) 标记为可疑")
        
        return df
    
    def get_labeling_rules(self) -> Dict[str, List[str]]:
        """
        获取标签规则描述
        
        Returns:
            规则描述字典
        """
        if self.platform == 'amazon':
            return {
                'platform': 'amazon',
                'rules': [
                    '未验证购买 (verified=False) +2分',
                    '评分极端 (1星或5星) +1分',
                    '评论长度异常 (<30或>2000字符) +1分',
                    '评论爆发性高 (<1天间隔) +2分',
                    '评分偏差大 (>1.5) +1分',
                    '用户评论数过多 (>100) +1分'
                ],
                'threshold': '>=4分标记为可疑'
            }
        elif self.platform == 'yelp':
            return {
                'platform': 'yelp',
                'rules': [
                    '评分极端 (1星或5星) +1分',
                    '评论长度异常 (<30或>2000字符) +1分',
                    '评论爆发性高 (<1天间隔) +2分',
                    '评分熵低 (<0.5) +2分',
                    '第一人称代词使用异常 (>0.15或<0.02) +1分',
                    '情感极端 (|sentiment|>0.8) +1分'
                ],
                'threshold': '>=4分标记为可疑'
            }
        else:
            return {'platform': self.platform, 'rules': [], 'threshold': 'N/A'}

