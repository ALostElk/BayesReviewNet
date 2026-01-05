#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯网络变量定义
定义所有随机变量及其取值空间
"""
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class BayesianVariable:
    """
    贝叶斯网络随机变量
    
    Attributes:
        name: 变量名
        states: 可能的取值（离散状态）
        description: 变量描述
        variable_type: 变量类型（'text', 'behavior', 'platform', 'label'）
    """
    name: str
    states: List[str]
    description: str
    variable_type: str


# ============ 文本特征变量 ============

REVIEW_LENGTH = BayesianVariable(
    name='review_length_discrete',
    states=['SHORT', 'NORMAL', 'LONG'],
    description='评论长度',
    variable_type='text'
)

SENTIMENT = BayesianVariable(
    name='sentiment_score_discrete',
    states=['NEGATIVE', 'NEUTRAL', 'POSITIVE'],
    description='情感倾向',
    variable_type='text'
)

SUBJECTIVITY = BayesianVariable(
    name='subjectivity_score_discrete',
    states=['OBJECTIVE', 'MIXED', 'SUBJECTIVE'],
    description='主观性',
    variable_type='text'
)


# ============ 用户行为变量 ============

USER_REVIEW_COUNT = BayesianVariable(
    name='user_review_count_discrete',
    states=['LOW', 'MEDIUM', 'HIGH'],
    description='用户评论数',
    variable_type='behavior'
)

RATING_DEVIATION = BayesianVariable(
    name='user_rating_deviation_discrete',
    states=['LOW', 'MEDIUM', 'HIGH'],
    description='评分偏差',
    variable_type='behavior'
)

RATING_ENTROPY = BayesianVariable(
    name='user_rating_entropy_discrete',
    states=['LOW', 'MEDIUM', 'HIGH'],
    description='评分熵（多样性）',
    variable_type='behavior'
)

REVIEW_BURSTINESS = BayesianVariable(
    name='user_review_burstiness_discrete',
    states=['RAPID', 'NORMAL', 'SPARSE'],
    description='评论爆发性',
    variable_type='behavior'
)


# ============ 平台特定变量 ============

PLATFORM = BayesianVariable(
    name='platform',
    states=['amazon', 'yelp'],
    description='评论来源平台（用于建模先验差异）',
    variable_type='platform'
)

VERIFIED_STATUS = BayesianVariable(
    name='verified',
    states=[True, False],
    description='验证购买状态（Amazon）',
    variable_type='platform'
)


# ============ 标签变量 ============

WEAK_LABEL = BayesianVariable(
    name='weak_label',
    states=[0, 1],  # 0=真实, 1=虚假
    description='弱标签（目标变量）',
    variable_type='label'
)


def get_all_variables() -> Dict[str, BayesianVariable]:
    """
    获取所有定义的变量
    
    Returns:
        变量名到变量对象的映射
    """
    return {
        'review_length': REVIEW_LENGTH,
        'sentiment': SENTIMENT,
        'subjectivity': SUBJECTIVITY,
        'user_review_count': USER_REVIEW_COUNT,
        'rating_deviation': RATING_DEVIATION,
        'rating_entropy': RATING_ENTROPY,
        'review_burstiness': REVIEW_BURSTINESS,
        'platform': PLATFORM,
        'verified_status': VERIFIED_STATUS,
        'weak_label': WEAK_LABEL
    }


def get_variables_by_type(variable_type: str) -> List[BayesianVariable]:
    """
    按类型获取变量
    
    Args:
        variable_type: 变量类型
        
    Returns:
        该类型的所有变量
    """
    all_vars = get_all_variables()
    return [var for var in all_vars.values() if var.variable_type == variable_type]

