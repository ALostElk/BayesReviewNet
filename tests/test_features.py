#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试特征工程模块
"""
import unittest
import pandas as pd
import numpy as np
from src.features.text_features import TextFeatureExtractor
from src.features.behavior_features import BehaviorFeatureExtractor


class TestTextFeatures(unittest.TestCase):
    """测试文本特征提取"""
    
    def setUp(self):
        """准备测试数据"""
        self.extractor = TextFeatureExtractor()
        self.test_text = "This is a great product! I love it so much."
    
    def test_extract_single(self):
        """测试单条文本特征提取"""
        features = self.extractor._extract_single(self.test_text)
        
        self.assertIn('review_length', features)
        self.assertIn('sentiment_score', features)
        self.assertIn('word_count', features)
        
        self.assertGreater(features['review_length'], 0)
        self.assertGreater(features['word_count'], 0)


class TestBehaviorFeatures(unittest.TestCase):
    """测试行为特征提取"""
    
    def setUp(self):
        """准备测试数据"""
        self.extractor = BehaviorFeatureExtractor()
        
        # 创建测试DataFrame
        self.df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2'],
            'rating': [5, 4, 3],
            'review_length': [100, 150, 200]
        })
    
    def test_compute_user_features(self):
        """测试用户特征计算"""
        user_features = self.extractor._compute_user_features(self.df)
        
        self.assertIn('user_review_count', user_features.columns)
        self.assertIn('user_avg_rating', user_features.columns)
        
        # user1应该有2条评论
        self.assertEqual(user_features.loc['user1', 'user_review_count'], 2)


if __name__ == '__main__':
    unittest.main()

