#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速入门示例
演示如何使用处理后的数据
"""
import pandas as pd
import os

def load_processed_data(dataset_name: str):
    """
    加载处理后的数据
    
    Args:
        dataset_name: 数据集名称 ('amazon', 'opspam', 'yelp')
    """
    # 加载最终数据
    data_path = f"discretized/{dataset_name}_final.parquet"
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在 {data_path}")
        print("请先运行 main.py 处理数据")
        return None
    
    df = pd.read_parquet(data_path)
    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*80}")
    print(f"总记录数: {len(df)}")
    print(f"\n列名: {list(df.columns)}")
    
    # 显示弱标签分布
    if 'weak_label' in df.columns:
        print(f"\n弱标签分布:")
        print(df['weak_label'].value_counts())
        print(f"\n可疑评论比例: {df['weak_label'].mean():.2%}")
    
    # 显示离散化特征示例
    discrete_cols = [col for col in df.columns if col.endswith('_discrete')]
    if discrete_cols:
        print(f"\n离散化特征 (共{len(discrete_cols)}个):")
        for col in discrete_cols[:5]:  # 显示前5个
            print(f"\n{col}:")
            print(df[col].value_counts())
    
    # 显示数据样本
    print(f"\n数据样本 (前3行):")
    print(df.head(3).to_string())
    
    return df


def demonstrate_bayesian_network_preparation(df: pd.DataFrame):
    """
    演示如何准备贝叶斯网络所需的数据
    
    Args:
        df: 处理后的DataFrame
    """
    print(f"\n{'='*80}")
    print("贝叶斯网络数据准备示例")
    print(f"{'='*80}")
    
    # 选择离散化特征
    discrete_features = [col for col in df.columns if col.endswith('_discrete')]
    
    if not discrete_features:
        print("没有找到离散化特征")
        return
    
    # 创建贝叶斯网络输入数据
    bn_data = df[discrete_features + ['weak_label']].copy()
    
    # 移除缺失值
    bn_data = bn_data.dropna()
    
    print(f"\n贝叶斯网络输入数据:")
    print(f"  - 特征数: {len(discrete_features)}")
    print(f"  - 样本数: {len(bn_data)}")
    print(f"  - 特征列表: {discrete_features[:5]}...")  # 显示前5个
    
    print(f"\n各特征的状态空间:")
    for col in discrete_features[:3]:  # 显示前3个特征的状态
        unique_values = bn_data[col].unique()
        print(f"  - {col}: {list(unique_values)}")
    
    print(f"\n数据样本:")
    print(bn_data.head(5).to_string())
    
    # 保存为CSV供贝叶斯网络工具使用
    output_path = "bn_input_data.csv"
    bn_data.to_csv(output_path, index=False)
    print(f"\n贝叶斯网络输入数据已保存到: {output_path}")


def main():
    """主函数"""
    print("="*80)
    print("贝叶斯网络虚假评论识别 - 快速入门示例")
    print("="*80)
    
    # 尝试加载Amazon数据
    df = load_processed_data('amazon')
    
    if df is not None:
        # 演示贝叶斯网络准备
        demonstrate_bayesian_network_preparation(df)
    
    print(f"\n{'='*80}")
    print("提示:")
    print("  1. 如果数据文件不存在，请先运行: python main.py --datasets all --sample")
    print("  2. 处理后的数据位于 discretized/ 目录")
    print("  3. 元数据和规则位于 metadata/ 目录")
    print("  4. 可以使用 pandas 直接加载 .parquet 或 .csv 文件")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

