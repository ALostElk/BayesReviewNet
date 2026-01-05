#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试集处理
OpSpam和YelpChi仅用于评估，不参与训练
"""
import pandas as pd
from typing import Dict

from src.utils.logging import setup_logger

logger = setup_logger("test_sets")


class OpSpamTestSet:
    """
    OpSpam测试集
    
    仅用于事后验证，具有ground truth标签
    """
    
    def __init__(self, data_path: str):
        """
        初始化测试集
        
        Args:
            data_path: 数据路径
        """
        self.data_path = data_path
        self.df = None
        logger.info(f"初始化OpSpam测试集: {data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        加载测试集
        
        Returns:
            测试集DataFrame
        """
        logger.info("加载OpSpam测试集...")
        
        if self.data_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path}")
        
        logger.info(f"OpSpam测试集加载完成，共 {len(self.df)} 条记录")
        return self.df
    
    def get_ground_truth(self) -> pd.Series:
        """
        获取ground truth标签
        
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
        
        return {
            'total_samples': len(self.df),
            'fake_samples': int(self.df['weak_label'].sum()),
            'real_samples': int((self.df['weak_label'] == 0).sum()),
            'fake_ratio': float(self.df['weak_label'].mean()),
            'sources': self.df['generation_source'].value_counts().to_dict() if 'generation_source' in self.df.columns else {}
        }


class YelpTestSet:
    """
    Yelp测试集
    
    使用平台过滤标签作为弱监督信号（noisy）
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
        
        return {
            'total_samples': len(self.df),
            'suspicious_samples': int(self.df['weak_label'].sum()) if 'weak_label' in self.df.columns else 0,
            'suspicious_ratio': float(self.df['weak_label'].mean()) if 'weak_label' in self.df.columns else 0.0,
            'label_source': self.df['label_source'].value_counts().to_dict() if 'label_source' in self.df.columns else {}
        }

