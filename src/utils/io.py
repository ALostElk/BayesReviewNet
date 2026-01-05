#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
输入输出工具
"""
import pandas as pd
import yaml
from typing import Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    """
    加载数据文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        DataFrame
    """
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    保存数据文件
    
    Args:
        df: DataFrame
        file_path: 文件路径
    """
    if file_path.endswith('.parquet'):
        df.to_parquet(file_path, index=False)
    elif file_path.endswith('.csv'):
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    """
    保存元数据到YAML文件
    
    Args:
        metadata: 元数据字典
        output_path: 输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False)

