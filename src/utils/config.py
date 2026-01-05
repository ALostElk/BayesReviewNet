#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置工具
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    if directory:  # 防止空字符串
        Path(directory).mkdir(parents=True, exist_ok=True)

