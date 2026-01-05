#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块
提供日志、配置加载等通用功能
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
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


def setup_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    
    # 文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log"),
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level))
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    """
    保存元数据到YAML文件
    
    Args:
        metadata: 元数据字典
        output_path: 输出路径
    """
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False)

