#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用预处理函数
包含时间标准化、文本清洗等工具函数
"""
import hashlib
import pandas as pd
from datetime import datetime
from typing import Optional


def standardize_timestamp(value, unit: str = 's') -> pd.Timestamp:
    """
    标准化时间戳
    
    Args:
        value: 时间值（Unix时间戳或字符串）
        unit: 时间单位 ('s', 'ms', 'ns')
        
    Returns:
        标准化的Timestamp对象
    """
    if pd.isna(value):
        return pd.NaT
    
    try:
        if isinstance(value, (int, float)):
            return pd.to_datetime(value, unit=unit, errors='coerce')
        elif isinstance(value, str):
            return pd.to_datetime(value, errors='coerce')
        else:
            return pd.NaT
    except:
        return pd.NaT


def clean_text(text: Optional[str]) -> Optional[str]:
    """
    清洗文本
    
    Args:
        text: 原始文本
        
    Returns:
        清洗后的文本
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    # 移除多余空白
    text = ' '.join(text.split())
    
    # 移除控制字符
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    return text.strip() if text.strip() else None


def generate_review_id(user_id: str, item_id: str, timestamp) -> str:
    """
    生成唯一的评论ID
    
    Args:
        user_id: 用户ID
        item_id: 商品ID
        timestamp: 时间戳
        
    Returns:
        MD5哈希的评论ID
    """
    content = f"{user_id}_{item_id}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()

