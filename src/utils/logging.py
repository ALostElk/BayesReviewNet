#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志工具
"""
import os
import logging
from pathlib import Path


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

