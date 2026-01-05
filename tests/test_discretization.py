#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试离散化模块
"""
import unittest
import pandas as pd
import numpy as np
from src.features.discretize import FeatureDiscretizer


class TestDiscretization(unittest.TestCase):
    """测试特征离散化"""
    
    def setUp(self):
        """准备测试数据"""
        self.config = {
            'review_length': {
                'bins': [0, 50, 200, 10000],
                'labels': ['SHORT', 'NORMAL', 'LONG']
            }
        }
        self.discretizer = FeatureDiscretizer(self.config)
        
        # 创建测试DataFrame
        self.df = pd.DataFrame({
            'review_length': [30, 100, 500]
        })
    
    def test_discretize(self):
        """测试离散化"""
        df_discretized = self.discretizer.discretize(self.df)
        
        self.assertIn('review_length_discrete', df_discretized.columns)
        
        # 检查离散化结果
        self.assertEqual(df_discretized.loc[0, 'review_length_discrete'], 'SHORT')
        self.assertEqual(df_discretized.loc[1, 'review_length_discrete'], 'NORMAL')
        self.assertEqual(df_discretized.loc[2, 'review_length_discrete'], 'LONG')


if __name__ == '__main__':
    unittest.main()

