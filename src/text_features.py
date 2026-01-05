#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本特征提取模块
基于统计和词典方法提取可解释的文本特征
不使用深度学习或预训练模型
"""
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from typing import Dict
from tqdm import tqdm

from src.utils import setup_logger

logger = setup_logger("text_features")

# 第一人称代词列表
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'
}


class TextFeatureExtractor:
    """文本特征提取器"""
    
    def __init__(self):
        self.feature_names = [
            'review_length',
            'sentiment_score',
            'subjectivity_score',
            'exclamation_ratio',
            'first_person_pronoun_ratio',
            'capital_letter_ratio',
            'word_count',
            'sentence_count',
            'avg_word_length',
            'unique_word_ratio'
        ]
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从DataFrame中提取文本特征
        
        Args:
            df: 包含review_text列的DataFrame
            
        Returns:
            添加了文本特征的DataFrame
        """
        logger.info("开始提取文本特征...")
        
        df = df.copy()
        
        # 初始化特征列
        for feature_name in self.feature_names:
            df[feature_name] = np.nan
        
        # 逐行提取特征
        for idx in tqdm(df.index, desc="提取文本特征"):
            text = df.loc[idx, 'review_text']
            
            # 跳过空文本
            if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
                continue
            
            try:
                features = self._extract_single_review(text)
                for feature_name, value in features.items():
                    df.loc[idx, feature_name] = value
            except Exception as e:
                logger.warning(f"提取特征失败 (索引 {idx}): {e}")
                continue
        
        logger.info("文本特征提取完成")
        return df
    
    def _extract_single_review(self, text: str) -> Dict[str, float]:
        """
        提取单条评论的特征
        
        Args:
            text: 评论文本
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基础统计特征
        features['review_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # 情感和主观性分析
        try:
            blob = TextBlob(text)
            features['sentiment_score'] = blob.sentiment.polarity  # [-1, 1]
            features['subjectivity_score'] = blob.sentiment.subjectivity  # [0, 1]
            features['sentence_count'] = len(blob.sentences)
        except:
            features['sentiment_score'] = 0.0
            features['subjectivity_score'] = 0.5
            features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        # 感叹号比例
        exclamation_count = text.count('!')
        features['exclamation_ratio'] = exclamation_count / len(text) if len(text) > 0 else 0.0
        
        # 第一人称代词比例
        words = text.lower().split()
        first_person_count = sum(1 for word in words if word in FIRST_PERSON_PRONOUNS)
        features['first_person_pronoun_ratio'] = first_person_count / len(words) if len(words) > 0 else 0.0
        
        # 大写字母比例
        capital_count = sum(1 for c in text if c.isupper())
        features['capital_letter_ratio'] = capital_count / len(text) if len(text) > 0 else 0.0
        
        # 平均词长
        if len(words) > 0:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0.0
        
        # 唯一词比例
        unique_words = set(words)
        features['unique_word_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0.0
        
        return features
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        获取特征描述
        
        Returns:
            特征名称到描述的映射
        """
        return {
            'review_length': '评论字符数',
            'sentiment_score': '情感分数 [-1=负面, 1=正面]',
            'subjectivity_score': '主观性分数 [0=客观, 1=主观]',
            'exclamation_ratio': '感叹号密度',
            'first_person_pronoun_ratio': '第一人称代词比例',
            'capital_letter_ratio': '大写字母比例',
            'word_count': '词数',
            'sentence_count': '句子数',
            'avg_word_length': '平均词长',
            'unique_word_ratio': '唯一词比例'
        }

