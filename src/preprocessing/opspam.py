#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpSpam数据预处理器
仅负责数据加载与字段标准化
注意：OpSpam仅用于测试，不参与训练
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.logging import setup_logger
from src.preprocessing.common import clean_text

logger = setup_logger("opspam_preprocessor")


class OpSpamPreprocessor:
    """OpSpam评论数据预处理器"""
    
    # 数据源配置（文件夹名 -> 标签）
    SOURCE_CONFIG = [
        ('generated_GPT3_positive', 1, 'gpt3', 'positive'),
        ('generated_GPT3_negative', 1, 'gpt3', 'negative'),
        ('generated_LLama_positive', 1, 'llama', 'positive'),
        ('generated_LLama_negative', 1, 'llama', 'negative'),
    ]
    
    def __init__(self, raw_data_dir: str):
        """
        初始化预处理器
        
        Args:
            raw_data_dir: 原始数据目录
        """
        self.raw_data_dir = Path(raw_data_dir)
        logger.info(f"初始化OpSpam预处理器，数据目录: {raw_data_dir}")
    
    def load_and_standardize(self) -> pd.DataFrame:
        """
        加载并标准化OpSpam数据
        
        Returns:
            标准化的DataFrame
        """
        logger.info("开始加载OpSpam原始数据...")
        
        all_reviews = []
        
        for folder_name, label, source, sentiment in self.SOURCE_CONFIG:
            folder_path = self.raw_data_dir / folder_name
            
            if not folder_path.exists():
                logger.warning(f"文件夹不存在: {folder_path}")
                continue
            
            txt_files = list(folder_path.glob("*.txt"))
            
            for txt_file in tqdm(txt_files, desc=f"加载{folder_name}"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        review_text = f.read().strip()
                    
                    all_reviews.append({
                        'review_text': review_text,
                        'label': label,  # 1表示虚假
                        'source': source,
                        'filename': txt_file.name,
                        'sentiment': sentiment
                    })
                except Exception as e:
                    logger.warning(f"读取文件 {txt_file} 失败: {e}")
                    continue
        
        logger.info(f"共加载 {len(all_reviews)} 条OpSpam评论")
        
        # 转换为DataFrame并标准化
        df = pd.DataFrame(all_reviews)
        df_std = self._standardize_fields(df)
        
        logger.info(f"OpSpam数据标准化完成，共 {len(df_std)} 条记录")
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
        
        # OpSpam没有真实的用户和商品ID，使用文件名
        df_std['user_id'] = df['filename'].apply(lambda x: f"opspam_{x}")
        df_std['item_id'] = 'opspam_item'
        df_std['review_id'] = df['filename'].apply(lambda x: f"opspam_{x}")
        
        # OpSpam没有时间戳和评分
        df_std['timestamp'] = pd.NaT
        df_std['rating'] = np.nan
        
        # 文本清洗
        df_std['review_text'] = df['review_text'].apply(clean_text)
        
        # 平台标识
        df_std['platform'] = 'opspam'
        
        # OpSpam特有字段
        df_std['verified'] = np.nan
        df_std['vote'] = np.nan
        
        # 真实标签（ground truth，仅用于测试）
        df_std['weak_label'] = df['label']  # 1=虚假, 0=真实
        df_std['label_source'] = 'ground_truth'
        df_std['generation_source'] = df['source']
        df_std['sentiment_type'] = df['sentiment']
        
        return df_std

