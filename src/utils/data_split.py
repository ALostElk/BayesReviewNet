#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨域数据划分工具
用于将目标域数据划分为校准集和测试集
"""
import pandas as pd
import numpy as np
from typing import Tuple

from src.utils.logging import setup_logger

logger = setup_logger("data_split")


def split_for_calibration(
    df: pd.DataFrame,
    calibration_ratio: float = 0.10,
    stratify_by: str = 'weak_label',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将目标域数据划分为校准集和测试集
    
    校准集用于似然分布参数的校准，测试集用于最终评估
    
    Args:
        df: 目标域数据
        calibration_ratio: 校准集比例（默认10%）
        stratify_by: 分层采样的列（确保两个集合的标签分布一致）
        random_state: 随机种子
        
    Returns:
        (calibration_df, test_df)
    """
    logger.info(f"开始划分目标域数据: 校准集={calibration_ratio*100:.0f}%, "
                f"测试集={100-calibration_ratio*100:.0f}%")
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 如果指定了分层列且存在，进行分层采样
    if stratify_by and stratify_by in df.columns:
        # 按标签分组采样
        calibration_indices = []
        test_indices = []
        
        for label in df[stratify_by].unique():
            label_indices = df[df[stratify_by] == label].index.tolist()
            n_calibration = int(len(label_indices) * calibration_ratio)
            
            # 随机打乱
            np.random.shuffle(label_indices)
            
            # 划分
            calibration_indices.extend(label_indices[:n_calibration])
            test_indices.extend(label_indices[n_calibration:])
        
        calibration_df = df.loc[calibration_indices].copy()
        test_df = df.loc[test_indices].copy()
        
        logger.info(f"  分层采样完成（按 {stratify_by}）")
    else:
        # 简单随机采样
        indices = df.index.tolist()
        np.random.shuffle(indices)
        
        n_calibration = int(len(indices) * calibration_ratio)
        calibration_indices = indices[:n_calibration]
        test_indices = indices[n_calibration:]
        
        calibration_df = df.loc[calibration_indices].copy()
        test_df = df.loc[test_indices].copy()
        
        logger.info(f"  随机采样完成")
    
    # 输出统计信息
    logger.info(f"  校准集: {len(calibration_df)} 条")
    logger.info(f"  测试集: {len(test_df)} 条")
    
    if stratify_by and stratify_by in df.columns:
        logger.info(f"  校准集标签分布:")
        for label, count in calibration_df[stratify_by].value_counts().items():
            logger.info(f"    - {label}: {count} ({count/len(calibration_df)*100:.2f}%)")
        
        logger.info(f"  测试集标签分布:")
        for label, count in test_df[stratify_by].value_counts().items():
            logger.info(f"    - {label}: {count} ({count/len(test_df)*100:.2f}%)")
    
    return calibration_df, test_df


def validate_split(
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = 'weak_label'
) -> bool:
    """
    验证数据划分的合理性
    
    检查：
    1. 两个集合无重叠
    2. 标签分布相似
    3. 校准集不为空
    
    Args:
        calibration_df: 校准集
        test_df: 测试集
        label_col: 标签列名
        
    Returns:
        验证是否通过
    """
    # 检查重叠
    overlap = set(calibration_df.index) & set(test_df.index)
    if len(overlap) > 0:
        logger.error(f"❌ 校准集和测试集存在重叠（{len(overlap)} 条记录）")
        return False
    
    logger.info("✓ 校准集和测试集无重叠")
    
    # 检查校准集大小
    if len(calibration_df) == 0:
        logger.error("❌ 校准集为空")
        return False
    
    logger.info(f"✓ 校准集大小合理（{len(calibration_df)} 条）")
    
    # 检查标签分布相似性
    if label_col in calibration_df.columns and label_col in test_df.columns:
        calib_fraud_rate = calibration_df[label_col].mean()
        test_fraud_rate = test_df[label_col].mean()
        diff = abs(calib_fraud_rate - test_fraud_rate)
        
        if diff > 0.05:  # 差异超过5%
            logger.warning(f"⚠ 标签分布差异较大: "
                          f"校准集={calib_fraud_rate:.4f}, "
                          f"测试集={test_fraud_rate:.4f}, "
                          f"差异={diff:.4f}")
        else:
            logger.info(f"✓ 标签分布相似: "
                       f"校准集={calib_fraud_rate:.4f}, "
                       f"测试集={test_fraud_rate:.4f}")
    
    return True

