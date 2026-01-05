#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Amazon数据预处理器
仅负责数据加载与字段标准化
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from src.utils.logging import setup_logger
from src.preprocessing.common import standardize_timestamp, clean_text, generate_review_id

logger = setup_logger("amazon_preprocessor")


class AmazonPreprocessor:
    """Amazon评论数据预处理器"""
    
    # 标准化字段映射
    FIELD_MAPPING = {
        'reviewerID': 'user_id',
        'asin': 'item_id',
        'unixReviewTime': 'timestamp',
        'overall': 'rating',
        'reviewText': 'review_text',
        'verified': 'verified',
        'vote': 'vote'
    }
    
    def __init__(self, raw_data_dir: str):
        """
        初始化预处理器
        
        Args:
            raw_data_dir: 原始数据目录
        """
        self.raw_data_dir = Path(raw_data_dir)
        logger.info(f"初始化Amazon预处理器，数据目录: {raw_data_dir}")
    
    def load_and_standardize(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        加载并标准化Amazon数据
        
        Args:
            sample_size: 采样数量，None表示全量加载
            
        Returns:
            标准化的DataFrame
        """
        logger.info("开始加载Amazon原始数据...")
        
        # 加载所有JSON文件
        all_reviews = []
        json_files = list(self.raw_data_dir.glob("*.json"))
        
        for json_file in tqdm(json_files, desc="加载Amazon文件"):
            try:
                reviews = self._load_json_file(json_file)
                all_reviews.extend(reviews)
                
                # 如果达到采样数量，提前停止
                if sample_size and len(all_reviews) >= sample_size:
                    all_reviews = all_reviews[:sample_size]
                    break
            except Exception as e:
                logger.warning(f"读取文件 {json_file} 失败: {e}")
                continue
        
        logger.info(f"共加载 {len(all_reviews)} 条Amazon评论")
        
        # 转换为DataFrame并标准化
        df = pd.DataFrame(all_reviews)
        df_std = self._standardize_fields(df)
        
        logger.info(f"Amazon数据标准化完成，共 {len(df_std)} 条记录")
        return df_std
    
    def _load_json_file(self, file_path: Path) -> list:
        """
        加载单个JSON文件（每行一个JSON对象）
        
        Args:
            file_path: 文件路径
            
        Returns:
            评论列表
        """
        reviews = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    reviews.append(review)
                except json.JSONDecodeError:
                    continue
        return reviews
    
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
        df_std['user_id'] = df.get('reviewerID', pd.Series([np.nan] * len(df)))
        df_std['item_id'] = df.get('asin', pd.Series([np.nan] * len(df)))
        
        # 生成唯一review_id
        df_std['review_id'] = df.apply(
            lambda x: generate_review_id(
                x.get('reviewerID'), 
                x.get('asin'), 
                x.get('unixReviewTime')
            ),
            axis=1
        )
        
        # 时间戳标准化
        df_std['timestamp'] = df['unixReviewTime'].apply(
            lambda x: standardize_timestamp(x, unit='s')
        ) if 'unixReviewTime' in df.columns else pd.NaT
        
        # 评分标准化
        df_std['rating'] = pd.to_numeric(
            df.get('overall', pd.Series([np.nan] * len(df))), 
            errors='coerce'
        )
        
        # 文本清洗
        df_std['review_text'] = df['reviewText'].apply(clean_text) \
            if 'reviewText' in df.columns else None
        
        # 平台标识
        df_std['platform'] = 'amazon'
        
        # Amazon特有字段
        df_std['verified'] = df.get('verified', np.nan)
        df_std['vote'] = pd.to_numeric(
            df.get('vote', pd.Series([np.nan] * len(df))), 
            errors='coerce'
        )
        
        # 预留标签字段（后续填充）
        df_std['weak_label'] = np.nan
        df_std['label_source'] = 'none'
        
        return df_std

