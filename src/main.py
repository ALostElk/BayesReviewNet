#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主执行脚本 - Pipeline调度器
协调整个贝叶斯网络建模流程
"""
import argparse
from pathlib import Path

from src.utils.config import load_config, ensure_dir
from src.utils.logging import setup_logger
from src.utils.io import save_data, save_metadata

# Preprocessing
from src.preprocessing import AmazonPreprocessor, YelpPreprocessor, OpSpamPreprocessor

# Features
from src.features import TextFeatureExtractor, BehaviorFeatureExtractor, FeatureDiscretizer
from src.features.discretize import create_discretization_summary

# Bayes
from src.bayes import BayesianNetworkStructure, CPDLearner, BayesianInference

# Evaluation
from src.evaluation import evaluate_model, OpSpamTestSet

logger = setup_logger("main")


class BayesReviewNetPipeline:
    """
    贝叶斯评论网络Pipeline
    
    完整流程：
    1. 数据预处理（Preprocessing）
    2. 特征工程（Feature Engineering）
    3. 贝叶斯网络建模（Bayesian Network）
    4. 推断与评估（Inference & Evaluation）
    """
    
    def __init__(self, config_path: str):
        """
        初始化Pipeline
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        logger.info("="*80)
        logger.info("BayesReviewNet Pipeline 初始化")
        logger.info("="*80)
    
    def run(self, dataset_name: str, structure_type: str = 'default'):
        """
        运行完整Pipeline
        
        Args:
            dataset_name: 数据集名称 ('amazon', 'opspam', 'yelp')
            structure_type: 贝叶斯网络结构类型
        """
        logger.info(f"\n开始处理数据集: {dataset_name}\n")
        
        # ========== 阶段1: 数据预处理 ==========
        logger.info("【阶段1】数据预处理")
        df = self._preprocess(dataset_name)
        
        # ========== 阶段2: 特征工程 ==========
        logger.info("\n【阶段2】特征工程")
        df = self._extract_features(df)
        
        # ========== 阶段3: 贝叶斯网络建模 ==========
        logger.info("\n【阶段3】贝叶斯网络建模")
        structure, cpd_learner = self._build_bayesian_network(df, structure_type)
        
        # ========== 阶段4: 推断与评估 ==========
        logger.info("\n【阶段4】推断与评估")
        df = self._inference_and_evaluate(df, structure, cpd_learner, dataset_name)
        
        # ========== 保存最终结果 ==========
        self._save_results(df, dataset_name)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"数据集 {dataset_name} 处理完成！")
        logger.info(f"{'='*80}\n")
    
    def _preprocess(self, dataset_name: str):
        """阶段1: 数据预处理"""
        if dataset_name == 'amazon':
            preprocessor = AmazonPreprocessor(
                self.config['data_paths']['amazon']['raw_dir']
            )
            sample_size = self.config['sampling']['amazon_sample_size'] \
                if self.config['sampling']['enabled'] else None
            df = preprocessor.load_and_standardize(sample_size)
        
        elif dataset_name == 'opspam':
            preprocessor = OpSpamPreprocessor(
                self.config['data_paths']['opspam']['raw_dir']
            )
            df = preprocessor.load_and_standardize()
        
        elif dataset_name == 'yelp':
            preprocessor = YelpPreprocessor(
                self.config['data_paths']['yelp']['raw_dir']
            )
            sample_size = self.config['sampling']['yelp_sample_size'] \
                if self.config['sampling']['enabled'] else None
            df = preprocessor.load_and_standardize(sample_size)
        
        else:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        # 保存标准化数据
        output_dir = self.config['data_paths'][dataset_name]['processed_dir']
        ensure_dir(output_dir)
        save_data(df, f"{output_dir}/{dataset_name}_standardized.parquet")
        
        return df
    
    def _extract_features(self, df):
        """阶段2: 特征工程"""
        # 2.1 文本特征
        text_extractor = TextFeatureExtractor()
        df = text_extractor.extract(df)
        
        # 2.2 行为特征
        behavior_extractor = BehaviorFeatureExtractor()
        df = behavior_extractor.extract(df)
        
        # 2.3 特征离散化
        discretizer = FeatureDiscretizer(self.config['discretization'])
        df = discretizer.discretize(df)
        
        return df
    
    def _build_bayesian_network(self, df, structure_type: str):
        """阶段3: 贝叶斯网络建模"""
        # 3.1 定义DAG结构
        structure = BayesianNetworkStructure()
        structure.define_structure(structure_type)
        
        logger.info(f"DAG结构已定义: {len(structure.edges)} 条边")
        logger.info(f"拓扑排序: {structure.get_topological_order()}")
        
        # 3.2 学习CPD
        cpd_learner = CPDLearner(structure)
        cpd_learner.learn_cpds(df, smoothing=1.0)
        
        return structure, cpd_learner
    
    def _inference_and_evaluate(self, df, structure, cpd_learner, dataset_name: str):
        """阶段4: 推断与评估"""
        # 4.1 贝叶斯推断
        inference = BayesianInference(structure, cpd_learner)
        df = inference.infer_posterior(df, target_variable='weak_label')
        
        # 4.2 评估（仅OpSpam有ground truth）
        if dataset_name == 'opspam':
            evaluation_result = evaluate_model(df)
            logger.info(f"\n评估结果:")
            logger.info(f"  Precision: {evaluation_result['metrics']['precision']:.4f}")
            logger.info(f"  Recall: {evaluation_result['metrics']['recall']:.4f}")
            logger.info(f"  F1-Score: {evaluation_result['metrics']['f1']:.4f}")
            logger.info(f"  ROC-AUC: {evaluation_result['metrics'].get('roc_auc', 'N/A')}")
            
            # 保存评估结果
            ensure_dir(self.config['output']['metadata_dir'])
            save_metadata(
                evaluation_result,
                f"{self.config['output']['metadata_dir']}/{dataset_name}_evaluation.yaml"
            )
        
        return df
    
    def _save_results(self, df, dataset_name: str):
        """保存最终结果"""
        output_dir = self.config['output']['discretized_dir']
        ensure_dir(output_dir)
        
        # 保存Parquet和CSV
        save_data(df, f"{output_dir}/{dataset_name}_final.parquet")
        save_data(df, f"{output_dir}/{dataset_name}_final.csv")
        
        logger.info(f"最终数据已保存到 {output_dir}/")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='BayesReviewNet - 基于贝叶斯网络的虚假评论识别'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['amazon', 'opspam', 'yelp', 'all'],
        default=['all'],
        help='要处理的数据集'
    )
    parser.add_argument(
        '--structure',
        type=str,
        choices=['default', 'naive'],
        default='default',
        help='贝叶斯网络结构类型'
    )
    
    args = parser.parse_args()
    
    # 确定要处理的数据集
    if 'all' in args.datasets:
        datasets = ['amazon', 'opspam', 'yelp']
    else:
        datasets = args.datasets
    
    # 初始化Pipeline
    pipeline = BayesReviewNetPipeline(args.config)
    
    # 处理每个数据集
    for dataset_name in datasets:
        try:
            pipeline.run(dataset_name, args.structure)
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 时出错: {e}", exc_info=True)
            continue
    
    logger.info("\n🎉 所有任务完成！")


if __name__ == '__main__':
    main()

