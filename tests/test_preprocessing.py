#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据预处理模块
"""
import unittest
import pandas as pd
from src.preprocessing.common import standardize_timestamp, clean_text, generate_review_id


class TestPreprocessing(unittest.TestCase):
    """测试预处理函数"""
    
    def test_standardize_timestamp(self):
        """测试时间戳标准化"""
        # Unix timestamp
        ts = standardize_timestamp(1609459200, unit='s')
        self.assertIsNotNone(ts)
        
        # 字符串格式
        ts = standardize_timestamp('2021-01-01', unit='s')
        self.assertIsNotNone(ts)
    
    def test_clean_text(self):
        """测试文本清洗"""
        # 正常文本
        text = clean_text("This is a test review.")
        self.assertEqual(text, "This is a test review.")
        
        # 多余空格
        text = clean_text("  Too   many   spaces  ")
        self.assertEqual(text, "Too many spaces")
        
        # 空文本
        text = clean_text(None)
        self.assertIsNone(text)
    
    def test_generate_review_id(self):
        """测试评论ID生成"""
        review_id = generate_review_id("user1", "item1", 1609459200)
        self.assertIsInstance(review_id, str)
        self.assertEqual(len(review_id), 32)  # MD5哈希长度


if __name__ == '__main__':
    unittest.main()

