#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
弱标签构造模块
基于启发式规则和平台信号构造弱监督标签
"""
import pandas as pd
import numpy as np
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def construct_amazon_weak_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    为Amazon数据构造弱标签
    
    基于以下启发式规则:
    1. verified=False (未验证购买) → 可疑
    2. 评分极端 (1星或5星) → 可疑
    3. 评论长度异常 (过短或过长) → 可疑
    4. 用户行为异常 (评论数过多、评分偏差大) → 可疑
    
    Args:
        df: 包含特征的DataFrame
        
    Returns:
        添加了weak_label的DataFrame
    """
    logger.info("开始为Amazon数据构造弱标签...")
    
    df = df.copy()
    
    # 初始化suspicion_score (可疑度分数)
    suspicion_score = np.zeros(len(df))
    
    # 规则1: 未验证购买 (+2分)
    if 'verified' in df.columns:
        suspicion_score += (~df['verified']).astype(int) * 2
    
    # 规则2: 极端评分 (+1分)
    if 'rating' in df.columns:
        extreme_ratings = (df['rating'] == 1) | (df['rating'] == 5)
        suspicion_score += extreme_ratings.astype(int)
    
    # 规则3: 评论长度异常 (+1分)
    if 'review_length' in df.columns:
        length_q25 = df['review_length'].quantile(0.25)
        length_q75 = df['review_length'].quantile(0.75)
        abnormal_length = (df['review_length'] < length_q25 * 0.5) | (df['review_length'] > length_q75 * 1.5)
        suspicion_score += abnormal_length.astype(int)
    
    # 规则4: 用户评论数异常 (+1分)
    if 'user_review_count' in df.columns:
        high_count = df['user_review_count'] > df['user_review_count'].quantile(0.9)
        suspicion_score += high_count.astype(int)
    
    # 规则5: 评分偏差大 (+1分)
    if 'user_rating_deviation' in df.columns:
        high_deviation = df['user_rating_deviation'] > df['user_rating_deviation'].quantile(0.75)
        suspicion_score += high_deviation.astype(int)
    
    # 转换为二值标签: suspicion_score >= 3 → 虚假(1), 否则 → 真实(0)
    df['weak_label'] = (suspicion_score >= 3).astype(int)
    df['label_source'] = 'heuristic'
    df['suspicion_score'] = suspicion_score  # 保留可疑度分数用于分析
    
    # 统计信息
    fake_count = (df['weak_label'] == 1).sum()
    real_count = (df['weak_label'] == 0).sum()
    fake_ratio = fake_count / len(df) * 100
    
    logger.info(f"Amazon弱标签构造完成:")
    logger.info(f"  - 真实: {real_count} ({100-fake_ratio:.2f}%)")
    logger.info(f"  - 虚假: {fake_count} ({fake_ratio:.2f}%)")
    
    return df


def construct_yelp_weak_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    为Yelp数据构造弱标签
    
    Yelp数据集的推荐过滤标签来自平台的过滤算法,但这是一个noisy label。
    如果原始数据没有filter标签,则使用类似Amazon的启发式规则。
    
    Args:
        df: 包含特征的DataFrame
        
    Returns:
        添加了weak_label的DataFrame
    """
    logger.info("开始为Yelp数据构造弱标签...")
    
    df = df.copy()
    
    # 检查是否有平台过滤标签 (原始Yelp数据可能没有这个字段)
    if 'filtered' in df.columns:
        # 使用平台过滤标签: filtered=True → 虚假(1), filtered=False → 真实(0)
        df['weak_label'] = df['filtered'].fillna(False).astype(int)
        df['label_source'] = 'platform'
        
        fake_count = (df['weak_label'] == 1).sum()
        real_count = (df['weak_label'] == 0).sum()
        fake_ratio = fake_count / len(df) * 100
        
        logger.info(f"Yelp弱标签构造完成(使用平台标签):")
        logger.info(f"  - 真实: {real_count} ({100-fake_ratio:.2f}%)")
        logger.info(f"  - 虚假: {fake_count} ({fake_ratio:.2f}%)")
    else:
        # 没有平台标签,使用启发式规则
        logger.info("Yelp数据无平台过滤标签,使用启发式规则...")
        
        suspicion_score = np.zeros(len(df))
        
        # 规则1: 极端评分 (+2分)
        if 'rating' in df.columns:
            extreme_ratings = (df['rating'] == 1) | (df['rating'] == 5)
            suspicion_score += extreme_ratings.astype(int) * 2
        
        # 规则2: 评论长度异常 (+1分)
        if 'review_length' in df.columns:
            length_q25 = df['review_length'].quantile(0.25)
            length_q75 = df['review_length'].quantile(0.75)
            abnormal_length = (df['review_length'] < length_q25 * 0.5) | (df['review_length'] > length_q75 * 1.5)
            suspicion_score += abnormal_length.astype(int)
        
        # 规则3: 用户评论数异常 (+1分)
        if 'user_review_count' in df.columns:
            high_count = df['user_review_count'] > df['user_review_count'].quantile(0.9)
            suspicion_score += high_count.astype(int)
        
        # 规则4: 评分偏差大 (+1分)
        if 'user_rating_deviation' in df.columns:
            high_deviation = df['user_rating_deviation'] > df['user_rating_deviation'].quantile(0.75)
            suspicion_score += high_deviation.astype(int)
        
        # 规则5: 评论爆发性异常 (+1分)
        if 'user_review_burstiness' in df.columns:
            rapid_burst = df['user_review_burstiness'] < df['user_review_burstiness'].quantile(0.25)
            suspicion_score += rapid_burst.astype(int)
        
        # 转换为二值标签: suspicion_score >= 3 → 虚假(1), 否则 → 真实(0)
        df['weak_label'] = (suspicion_score >= 3).astype(int)
        df['label_source'] = 'heuristic'
        df['suspicion_score'] = suspicion_score
        
        fake_count = (df['weak_label'] == 1).sum()
        real_count = (df['weak_label'] == 0).sum()
        fake_ratio = fake_count / len(df) * 100
        
        logger.info(f"Yelp弱标签构造完成(使用启发式规则):")
        logger.info(f"  - 真实: {real_count} ({100-fake_ratio:.2f}%)")
        logger.info(f"  - 虚假: {fake_count} ({fake_ratio:.2f}%)")
    
    return df


def construct_weak_label(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    根据平台选择合适的弱标签构造方法
    
    Args:
        df: 包含特征的DataFrame
        platform: 平台名称 ('amazon' or 'yelp')
        
    Returns:
        添加了weak_label的DataFrame
    """
    if platform.lower() == 'amazon':
        return construct_amazon_weak_label(df)
    elif platform.lower() == 'yelp':
        return construct_yelp_weak_label(df)
    else:
        raise ValueError(f"不支持的平台: {platform}")

