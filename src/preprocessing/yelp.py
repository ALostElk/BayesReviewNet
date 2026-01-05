#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yelp数据预处理器
仅负责数据加载与字段标准化
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from src.utils.logging import setup_logger
from src.preprocessing.common import standardize_timestamp, clean_text

logger = setup_logger("yelp_preprocessor")


class YelpPreprocessor:
    """Yelp评论数据预处理器"""
    
    def __init__(self, raw_data_dir: str):
        """
        初始化预处理器
        
        Args:
            raw_data_dir: 原始数据目录
        """
        self.raw_data_dir = Path(raw_data_dir)
        logger.info(f"初始化Yelp预处理器，数据目录: {raw_data_dir}")
    
    def load_and_standardize(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        加载并标准化Yelp数据
        
        Args:
            sample_size: 采样数量，None表示全量加载
            
        Returns:
            标准化的DataFrame
        """
        logger.info("开始加载Yelp原始数据...")
        
        # 加载评论数据
        review_file = self.raw_data_dir / "yelp_academic_dataset_review.json"
        reviews = []
        
        with open(review_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载Yelp评论"):
                try:
                    review = json.loads(line.strip())
                    reviews.append(review)
                    
                    if sample_size and len(reviews) >= sample_size:
                        break
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"共加载 {len(reviews)} 条Yelp评论")
        
        # 转换为DataFrame并标准化
        df = pd.DataFrame(reviews)
        df_std = self._standardize_fields(df)
        
        logger.info(f"Yelp数据标准化完成，共 {len(df_std)} 条记录")
        return df_std
    
    def _standardize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化字段名和格式
        
        Args:
            df: 原始DataFrame
            
        Returns:
            标准化的DataFrame
        """
        df_std = pd.DataFrame()
        
        # 基础字段映射
        df_std['user_id'] = df.get('user_id', np.nan)
        df_std['item_id'] = df.get('business_id', np.nan)
        df_std['review_id'] = df.get('review_id', np.nan)
        
        # 时间戳标准化
        df_std['timestamp'] = df['date'].apply(
            lambda x: standardize_timestamp(x)
        ) if 'date' in df.columns else pd.NaT
        
        # 评分标准化
        df_std['rating'] = pd.to_numeric(
            df.get('stars', np.nan), 
            errors='coerce'
        )
        
        # 文本清洗
        df_std['review_text'] = df['text'].apply(clean_text) \
            if 'text' in df.columns else None
        
        # 平台标识
        df_std['platform'] = 'yelp'
        
        # Yelp特有字段
        df_std['verified'] = np.nan  # Yelp没有verified字段
        df_std['vote'] = pd.to_numeric(df.get('useful', np.nan), errors='coerce')
        df_std['funny'] = pd.to_numeric(df.get('funny', np.nan), errors='coerce')
        df_std['cool'] = pd.to_numeric(df.get('cool', np.nan), errors='coerce')
        
        # 预留标签字段
        df_std['weak_label'] = np.nan
        df_std['label_source'] = 'none'
        
        return df_std

