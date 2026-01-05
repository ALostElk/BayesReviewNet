#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主执行脚本
协调整个数据处理流程
"""
import os
import argparse
from pathlib import Path

from src.utils import load_config, setup_logger, ensure_dir, save_metadata
from src.data_loader import load_dataset
from src.text_features import TextFeatureExtractor
from src.user_features import UserFeatureAggregator
from src.discretizer import FeatureDiscretizer, create_discretization_summary
from src.weak_labeler import WeakLabeler

logger = setup_logger("main")


def process_dataset(dataset_name: str, config: dict, args: argparse.Namespace):
    """
    处理单个数据集的完整流程
    
    Args:
        dataset_name: 数据集名称
        config: 配置字典
        args: 命令行参数
    """
    logger.info(f"{'='*80}")
    logger.info(f"开始处理数据集: {dataset_name}")
    logger.info(f"{'='*80}")
    
    # 1. 加载和标准化数据
    logger.info("步骤 1/5: 数据加载和标准化")
    sample_size = None
    if config['sampling']['enabled']:
        if dataset_name == 'amazon':
            sample_size = config['sampling']['amazon_sample_size']
        elif dataset_name == 'yelp':
            sample_size = config['sampling']['yelp_sample_size']
    
    df = load_dataset(dataset_name, config, sample_size)
    logger.info(f"数据加载完成，共 {len(df)} 条记录")
    
    # 保存标准化数据
    output_dir = config['data_paths'][dataset_name]['processed_dir']
    ensure_dir(output_dir)
    standardized_path = os.path.join(output_dir, f"{dataset_name}_standardized.parquet")
    df.to_parquet(standardized_path, index=False)
    logger.info(f"标准化数据已保存: {standardized_path}")
    
    # 2. 提取文本特征
    logger.info("步骤 2/5: 文本特征提取")
    text_extractor = TextFeatureExtractor()
    df = text_extractor.extract_features(df)
    
    # 保存文本特征描述
    text_feature_desc = text_extractor.get_feature_descriptions()
    text_desc_path = os.path.join(config['output']['metadata_dir'], f"{dataset_name}_text_features.yaml")
    ensure_dir(config['output']['metadata_dir'])
    save_metadata(text_feature_desc, text_desc_path)
    
    # 3. 聚合用户特征
    logger.info("步骤 3/5: 用户行为特征聚合")
    user_aggregator = UserFeatureAggregator()
    df = user_aggregator.aggregate_features(df)
    
    # 保存用户特征描述
    user_feature_desc = user_aggregator.get_feature_descriptions()
    user_desc_path = os.path.join(config['output']['metadata_dir'], f"{dataset_name}_user_features.yaml")
    save_metadata(user_feature_desc, user_desc_path)
    
    # 保存特征数据
    features_path = os.path.join(config['output']['features_dir'], f"{dataset_name}_features.parquet")
    ensure_dir(config['output']['features_dir'])
    df.to_parquet(features_path, index=False)
    logger.info(f"特征数据已保存: {features_path}")
    
    # 4. 变量离散化
    logger.info("步骤 4/5: 变量离散化")
    discretizer = FeatureDiscretizer(config['discretization'])
    df = discretizer.discretize(df)
    
    # 保存离散化规则
    discretization_rules = discretizer.get_discretization_rules()
    rules_path = os.path.join(config['output']['metadata_dir'], f"{dataset_name}_discretization_rules.json")
    discretizer.save_rules(rules_path)
    
    # 创建离散化摘要
    discretization_summary = create_discretization_summary(df, discretization_rules)
    summary_path = os.path.join(config['output']['metadata_dir'], f"{dataset_name}_discretization_summary.yaml")
    save_metadata(discretization_summary, summary_path)
    
    # 5. 构造弱标签
    logger.info("步骤 5/5: 弱标签构造")
    weak_labeler = WeakLabeler(dataset_name)
    df = weak_labeler.construct_weak_labels(df)
    
    # 保存标签规则
    labeling_rules = weak_labeler.get_labeling_rules()
    label_rules_path = os.path.join(config['output']['metadata_dir'], f"{dataset_name}_labeling_rules.yaml")
    save_metadata(labeling_rules, label_rules_path)
    
    # 保存最终数据
    final_path = os.path.join(config['output']['discretized_dir'], f"{dataset_name}_final.parquet")
    ensure_dir(config['output']['discretized_dir'])
    df.to_parquet(final_path, index=False)
    logger.info(f"最终数据已保存: {final_path}")
    
    # 保存CSV版本（便于查看）
    csv_path = os.path.join(config['output']['discretized_dir'], f"{dataset_name}_final.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"CSV版本已保存: {csv_path}")
    
    # 生成数据摘要
    summary = {
        'dataset': dataset_name,
        'total_records': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'weak_label_distribution': df['weak_label'].value_counts().to_dict() if 'weak_label' in df.columns else {},
        'output_files': {
            'standardized': standardized_path,
            'features': features_path,
            'final': final_path,
            'csv': csv_path
        }
    }
    summary_path = os.path.join(config['output']['metadata_dir'], f"{dataset_name}_summary.yaml")
    save_metadata(summary, summary_path)
    
    logger.info(f"数据集 {dataset_name} 处理完成！")
    logger.info(f"{'='*80}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='贝叶斯网络虚假评论识别 - 数据处理')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        choices=['amazon', 'opspam', 'yelp', 'all'],
                        default=['all'], help='要处理的数据集')
    parser.add_argument('--sample', action='store_true', help='启用采样模式')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 如果启用采样，更新配置
    if args.sample:
        config['sampling']['enabled'] = True
        logger.info("采样模式已启用")
    
    # 确定要处理的数据集
    if 'all' in args.datasets:
        datasets = ['amazon', 'opspam', 'yelp']
    else:
        datasets = args.datasets
    
    logger.info(f"将处理以下数据集: {', '.join(datasets)}")
    
    # 处理每个数据集
    for dataset_name in datasets:
        try:
            process_dataset(dataset_name, config, args)
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 时出错: {e}", exc_info=True)
            continue
    
    logger.info("所有数据集处理完成！")


if __name__ == '__main__':
    main()

