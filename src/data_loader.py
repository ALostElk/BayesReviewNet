#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载和标准化模块
负责加载各数据集并转换为统一格式
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import hashlib

from src.utils import setup_logger

logger = setup_logger("data_loader")


class AmazonDataLoader:
    """Amazon评论数据加载器"""
    
    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir
        
    def load_and_standardize(self, sample_size: int = None) -> pd.DataFrame:
        """
        加载并标准化Amazon数据
        
        Args:
            sample_size: 采样大小，None表示加载全部
            
        Returns:
            标准化的DataFrame
        """
        logger.info("开始加载Amazon数据集...")
        
        all_reviews = []
        json_files = list(Path(self.raw_dir).glob("*.json"))
        
        for json_file in tqdm(json_files, desc="加载Amazon文件"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            review = json.loads(line.strip())
                            all_reviews.append(review)
                        except json.JSONDecodeError:
                            continue
                            
                # 如果启用采样且已达到目标数量，提前退出
                if sample_size and len(all_reviews) >= sample_size:
                    all_reviews = all_reviews[:sample_size]
                    break
            except Exception as e:
                logger.warning(f"读取文件 {json_file} 失败: {e}")
                continue
        
        logger.info(f"共加载 {len(all_reviews)} 条Amazon评论")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_reviews)
        
        # 标准化字段名
        df_std = pd.DataFrame()
        df_std['user_id'] = df.get('reviewerID', pd.Series([np.nan] * len(df)))
        df_std['item_id'] = df.get('asin', pd.Series([np.nan] * len(df)))
        df_std['review_id'] = df.apply(
            lambda x: self._generate_review_id(x.get('reviewerID'), x.get('asin'), x.get('unixReviewTime')),
            axis=1
        )
        df_std['timestamp'] = pd.to_datetime(df.get('unixReviewTime', pd.Series([np.nan] * len(df))), unit='s', errors='coerce')
        df_std['rating'] = pd.to_numeric(df.get('overall', pd.Series([np.nan] * len(df))), errors='coerce')
        df_std['review_text'] = df.get('reviewText', pd.Series([np.nan] * len(df)))
        df_std['platform'] = 'amazon'
        
        # Amazon特有字段
        df_std['verified'] = df.get('verified', pd.Series([np.nan] * len(df)))
        df_std['vote'] = pd.to_numeric(df.get('vote', pd.Series([np.nan] * len(df))), errors='coerce')
        
        # 弱标签（后续构造）
        df_std['weak_label'] = np.nan
        df_std['label_source'] = 'none'
        
        logger.info(f"Amazon数据标准化完成，共 {len(df_std)} 条记录")
        return df_std
    
    @staticmethod
    def _generate_review_id(user_id, item_id, timestamp) -> str:
        """生成唯一的review_id"""
        content = f"{user_id}_{item_id}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()


class OpSpamDataLoader:
    """OpSpam数据加载器"""
    
    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir
        
    def load_and_standardize(self) -> pd.DataFrame:
        """
        加载并标准化OpSpam数据
        
        Returns:
            标准化的DataFrame
        """
        logger.info("开始加载OpSpam数据集...")
        
        all_reviews = []
        
        # 定义数据源和标签
        sources = [
            ('generated_GPT3_positive', 1, 'gpt3'),
            ('generated_GPT3_negative', 1, 'gpt3'),
            ('generated_LLama_positive', 1, 'llama'),
            ('generated_LLama_negative', 1, 'llama'),
        ]
        
        for folder_name, label, source in sources:
            folder_path = os.path.join(self.raw_dir, folder_name)
            if not os.path.exists(folder_path):
                logger.warning(f"文件夹不存在: {folder_path}")
                continue
                
            txt_files = list(Path(folder_path).glob("*.txt"))
            
            for txt_file in tqdm(txt_files, desc=f"加载{folder_name}"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        review_text = f.read().strip()
                        
                    all_reviews.append({
                        'review_text': review_text,
                        'label': label,  # 1表示虚假
                        'source': source,
                        'filename': txt_file.name,
                        'sentiment': 'positive' if 'positive' in folder_name else 'negative'
                    })
                except Exception as e:
                    logger.warning(f"读取文件 {txt_file} 失败: {e}")
                    continue
        
        logger.info(f"共加载 {len(all_reviews)} 条OpSpam评论")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_reviews)
        
        # 标准化字段名
        df_std = pd.DataFrame()
        df_std['user_id'] = df['filename'].apply(lambda x: f"opspam_{x}")
        df_std['item_id'] = 'opspam_item'  # OpSpam没有商品信息
        df_std['review_id'] = df['filename'].apply(lambda x: f"opspam_{x}")
        df_std['timestamp'] = pd.NaT  # OpSpam没有时间戳
        df_std['rating'] = np.nan  # OpSpam没有评分
        df_std['review_text'] = df['review_text']
        df_std['platform'] = 'opspam'
        
        # OpSpam特有字段
        df_std['verified'] = np.nan
        df_std['vote'] = np.nan
        
        # 真实标签（仅用于测试）
        df_std['weak_label'] = df['label']  # 1表示虚假
        df_std['label_source'] = 'ground_truth'
        df_std['generation_source'] = df['source']
        df_std['sentiment_type'] = df['sentiment']
        
        logger.info(f"OpSpam数据标准化完成，共 {len(df_std)} 条记录")
        return df_std


class YelpDataLoader:
    """Yelp数据加载器"""
    
    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir
        
    def load_and_standardize(self, sample_size: int = None) -> pd.DataFrame:
        """
        加载并标准化Yelp数据
        
        Args:
            sample_size: 采样大小，None表示加载全部
            
        Returns:
            标准化的DataFrame
        """
        logger.info("开始加载Yelp数据集...")
        
        # 加载评论数据
        review_file = os.path.join(self.raw_dir, "yelp_academic_dataset_review.json")
        user_file = os.path.join(self.raw_dir, "yelp_academic_dataset_user.json")
        
        reviews = []
        with open(review_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="加载Yelp评论")):
                try:
                    review = json.loads(line.strip())
                    reviews.append(review)
                    
                    if sample_size and len(reviews) >= sample_size:
                        break
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"共加载 {len(reviews)} 条Yelp评论")
        
        # 转换为DataFrame
        df = pd.DataFrame(reviews)
        
        # 标准化字段名
        df_std = pd.DataFrame()
        df_std['user_id'] = df.get('user_id', pd.Series([np.nan] * len(df)))
        df_std['item_id'] = df.get('business_id', pd.Series([np.nan] * len(df)))
        df_std['review_id'] = df.get('review_id', pd.Series([np.nan] * len(df)))
        df_std['timestamp'] = pd.to_datetime(df.get('date', pd.Series([np.nan] * len(df))), errors='coerce')
        df_std['rating'] = pd.to_numeric(df.get('stars', pd.Series([np.nan] * len(df))), errors='coerce')
        df_std['review_text'] = df.get('text', pd.Series([np.nan] * len(df)))
        df_std['platform'] = 'yelp'
        
        # Yelp特有字段
        df_std['verified'] = np.nan  # Yelp没有verified字段
        df_std['vote'] = pd.to_numeric(df.get('useful', pd.Series([np.nan] * len(df))), errors='coerce')
        df_std['funny'] = pd.to_numeric(df.get('funny', pd.Series([np.nan] * len(df))), errors='coerce')
        df_std['cool'] = pd.to_numeric(df.get('cool', pd.Series([np.nan] * len(df))), errors='coerce')
        
        # 弱标签（后续构造）
        df_std['weak_label'] = np.nan
        df_std['label_source'] = 'none'
        
        logger.info(f"Yelp数据标准化完成，共 {len(df_std)} 条记录")
        return df_std


def load_dataset(dataset_name: str, config: Dict, sample_size: int = None) -> pd.DataFrame:
    """
    统一数据加载接口
    
    Args:
        dataset_name: 数据集名称 ('amazon', 'opspam', 'yelp')
        config: 配置字典
        sample_size: 采样大小
        
    Returns:
        标准化的DataFrame
    """
    if dataset_name == 'amazon':
        loader = AmazonDataLoader(config['data_paths']['amazon']['raw_dir'])
        return loader.load_and_standardize(sample_size)
    elif dataset_name == 'opspam':
        loader = OpSpamDataLoader(config['data_paths']['opspam']['raw_dir'])
        return loader.load_and_standardize()
    elif dataset_name == 'yelp':
        loader = YelpDataLoader(config['data_paths']['yelp']['raw_dir'])
        return loader.load_and_standardize(sample_size)
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")

