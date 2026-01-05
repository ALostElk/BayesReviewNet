#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估指标
包含Precision, Recall, F1, ROC-AUC等
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.utils.logging import setup_logger

logger = setup_logger("metrics")


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（用于AUC）
        
    Returns:
        指标字典
    """
    metrics = {}
    
    # 基础分类指标
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC（如果提供概率）
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = np.nan
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    
    # 准确率
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return metrics


def evaluate_model(
    df: pd.DataFrame, 
    true_label_col: str = 'weak_label',
    prob_col: str = 'weak_label_posterior_prob',
    threshold: float = 0.5
) -> Dict:
    """
    评估模型性能
    
    Args:
        df: 包含真实标签和预测概率的DataFrame
        true_label_col: 真实标签列名
        prob_col: 预测概率列名
        threshold: 分类阈值
        
    Returns:
        评估结果字典
    """
    logger.info("开始模型评估...")
    
    # 过滤有效数据
    df_valid = df[[true_label_col, prob_col]].dropna()
    
    if len(df_valid) == 0:
        logger.warning("无有效评估数据")
        return {'error': '无有效数据'}
    
    y_true = df_valid[true_label_col].values
    y_prob = df_valid[prob_col].values
    
    # 根据阈值生成预测标签
    y_pred = (y_prob >= threshold).astype(int)
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # 生成分类报告
    try:
        report = classification_report(y_true, y_pred, target_names=['真实', '虚假'], output_dict=True)
    except:
        report = {}
    
    logger.info(f"评估完成 - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    return {
        'metrics': metrics,
        'classification_report': report,
        'threshold': threshold,
        'n_samples': len(df_valid)
    }


def find_optimal_threshold(
    df: pd.DataFrame,
    true_label_col: str = 'weak_label',
    prob_col: str = 'weak_label_posterior_prob',
    metric: str = 'f1'
) -> Dict:
    """
    寻找最优分类阈值
    
    Args:
        df: 数据DataFrame
        true_label_col: 真实标签列名
        prob_col: 预测概率列名
        metric: 优化指标 ('f1', 'precision', 'recall')
        
    Returns:
        最优阈值和对应指标
    """
    df_valid = df[[true_label_col, prob_col]].dropna()
    
    if len(df_valid) == 0:
        return {'error': '无有效数据'}
    
    y_true = df_valid[true_label_col].values
    y_prob = df_valid[prob_col].values
    
    # 尝试不同阈值
    thresholds = np.linspace(0.1, 0.9, 17)
    best_threshold = 0.5
    best_score = 0.0
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        score = metrics[metric]
        results.append({
            'threshold': threshold,
            'score': score,
            'metrics': metrics
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"最优阈值: {best_threshold:.2f}, {metric}: {best_score:.4f}")
    
    return {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'metric': metric,
        'all_results': results
    }

