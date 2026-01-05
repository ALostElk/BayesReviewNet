#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主执行脚本 - Pipeline调度器
协调整个贝叶斯网络建模流程
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_config, ensure_dir
from utils.logging import setup_logger
from utils.io import save_data, save_metadata

# Preprocessing
from preprocessing import AmazonPreprocessor, YelpPreprocessor

# Features
from features import TextFeatureExtractor, BehaviorFeatureExtractor, FeatureDiscretizer
from features.discretize import create_discretization_summary

# Bayes
from bayes import BayesianNetworkStructure, CPDLearner, BayesianInference

# Evaluation
from evaluation import evaluate_model

logger = setup_logger("main")


class BayesReviewNetPipeline:
    """
    贝叶斯评论网络Pipeline
    
    完整流程：
    1. 数据预处理（Preprocessing）- 支持Amazon和Yelp数据集
    2. 特征工程（Feature Engineering）- Text + Behavior + Network多视角特征
    3. 贝叶斯网络建模（Bayesian Network）- DAG结构与CPD学习
    4. 推断与评估（Inference & Evaluation）
    """
    
    # 支持的数据集
    SUPPORTED_DATASETS = ['amazon', 'yelp']
    
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
    
    def run(self, dataset_name: str, structure_type: str = 'default') -> dict:
        """
        运行完整Pipeline
        
        Args:
            dataset_name: 数据集名称 ('amazon', 'yelp')
            structure_type: 贝叶斯网络结构类型
            
        Returns:
            处理结果统计字典
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: {self.SUPPORTED_DATASETS}")
        
        logger.info(f"\n开始处理数据集: {dataset_name}\n")
        
        # ========== 阶段1: 数据预处理 ==========
        logger.info("【阶段1】数据预处理")
        df = self._preprocess(dataset_name)
        
        # ========== 阶段2: 特征工程 ==========
        logger.info("\n【阶段2】特征工程 - 提取Text + Behavior + Network特征")
        df = self._extract_features(df)
        
        # ========== 阶段2.5: 构造弱标签 ==========
        logger.info("\n【阶段2.5】弱标签构造")
        df = self._construct_weak_labels(df, dataset_name)
        
        # ========== 阶段3: 贝叶斯网络建模 ==========
        logger.info("\n【阶段3】贝叶斯网络建模")
        structure, cpd_learner = self._build_bayesian_network(df, structure_type)
        
        # ========== 阶段4: 推断与评估 ==========
        logger.info("\n【阶段4】推断与评估")
        df = self._inference_and_evaluate(df, structure, cpd_learner, dataset_name)
        
        # ========== 保存最终结果 ==========
        output_path = self._save_results(df, dataset_name)
        
        # ========== 生成统计信息 ==========
        stats = self._generate_statistics(df, dataset_name, output_path)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"数据集 {dataset_name} 处理完成！")
        logger.info(f"{'='*80}\n")
        
        return stats
    
    def _preprocess(self, dataset_name: str):
        """
        阶段1: 数据预处理
        
        所有数据集统一输出包含以下字段的DataFrame:
        - user_id, item_id, review_id, timestamp, rating, review_text
        - platform, verified, vote
        - weak_label, label_source
        """
        if dataset_name == 'amazon':
            preprocessor = AmazonPreprocessor(
                self.config['data_paths']['amazon']['raw_dir']
            )
            sample_size = self.config['sampling']['amazon_sample_size'] \
                if self.config['sampling']['enabled'] else None
            df = preprocessor.load_and_standardize(sample_size)
        
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
        
        logger.info(f"数据预处理完成: {len(df)} 条记录")
        return df
    
    def _extract_features(self, df):
        """
        阶段2: 特征工程
        
        提取多视角特征:
        - Text特征: 文本统计、情感、主观性等
        - Behavior特征: 用户评论数、评分模式、时间模式等
        - Network特征: 用户-商品图结构特征（未来扩展）
        """
        # 2.1 文本特征
        logger.info("  → 提取Text特征...")
        text_extractor = TextFeatureExtractor()
        df = text_extractor.extract(df)
        
        # 2.2 行为特征
        logger.info("  → 提取Behavior特征...")
        behavior_extractor = BehaviorFeatureExtractor()
        df = behavior_extractor.extract(df)
        
        # TODO: 2.3 网络特征（未来扩展）
        # logger.info("  → 提取Network特征...")
        # network_extractor = NetworkFeatureExtractor()
        # df = network_extractor.extract(df)
        
        # 2.4 特征离散化
        logger.info("  → 特征离散化...")
        discretizer = FeatureDiscretizer(self.config['discretization'])
        df = discretizer.discretize(df)
        
        logger.info(f"特征工程完成: {len([c for c in df.columns if '_discrete' in c])} 个离散特征")
        return df
    
    def _construct_weak_labels(self, df, dataset_name: str):
        """
        阶段2.5: 构造弱标签
        
        基于启发式规则或平台信号构造弱监督标签
        必须在特征提取之后、贝叶斯网络建模之前执行
        """
        from src.preprocessing.weak_labeling import construct_weak_label
        
        platform = df['platform'].iloc[0] if 'platform' in df.columns else dataset_name
        df = construct_weak_label(df, platform)
        
        return df
    
    def _build_bayesian_network(self, df, structure_type: str):
        """
        阶段3: 贝叶斯网络建模
        
        基于多视角特征构建贝叶斯网络
        """
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
        """
        阶段4: 推断与评估
        
        使用贝叶斯推断计算后验概率
        """
        # 4.1 贝叶斯推断
        inference = BayesianInference(structure, cpd_learner)
        df = inference.infer_posterior(df, target_variable='weak_label')
        
        # 4.2 如果有弱标签，可以进行评估
        if 'weak_label' in df.columns and df['weak_label'].notna().any():
            logger.info("\n评估模型性能...")
            try:
                evaluation_result = evaluate_model(df)
                logger.info(f"  Precision: {evaluation_result['metrics']['precision']:.4f}")
                logger.info(f"  Recall: {evaluation_result['metrics']['recall']:.4f}")
                logger.info(f"  F1-Score: {evaluation_result['metrics']['f1']:.4f}")
                logger.info(f"  ROC-AUC: {evaluation_result['metrics'].get('roc_auc', 'N/A')}")
            except Exception as e:
                logger.warning(f"评估失败: {e}")
        
        return df
    
    def _save_results(self, df, dataset_name: str) -> str:
        """
        保存最终结果到data/processed/目录
        
        Returns:
            输出文件路径
        """
        output_dir = self.config['data_paths'][dataset_name]['processed_dir']
        ensure_dir(output_dir)
        
        # 保存最终处理结果
        output_path = f"{output_dir}/{dataset_name}_final.parquet"
        save_data(df, output_path)
        
        logger.info(f"最终数据已保存到 {output_dir}/")
        return output_path
    
    def _generate_statistics(self, df, dataset_name: str, output_path: str) -> dict:
        """
        生成数据集处理统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'dataset': dataset_name,
            'total_samples': len(df),
            'output_file': output_path,
            'features': {}
        }
        
        # 统计文本特征
        text_features = [col for col in df.columns if col in [
            'review_length', 'sentiment_score', 'subjectivity_score',
            'exclamation_ratio', 'first_person_pronoun_ratio'
        ]]
        stats['features']['text'] = len(text_features)
        
        # 统计行为特征
        behavior_features = [col for col in df.columns if col.startswith('user_')]
        stats['features']['behavior'] = len(behavior_features)
        
        # 统计离散化特征
        discrete_features = [col for col in df.columns if col.endswith('_discrete')]
        stats['features']['discrete'] = len(discrete_features)
        
        # 统计弱标签分布（如果存在）
        if 'weak_label' in df.columns:
            label_dist = df['weak_label'].value_counts().to_dict()
            stats['weak_label_distribution'] = {
                'suspicious': int(label_dist.get(1, 0)),
                'normal': int(label_dist.get(0, 0)),
                'missing': int(df['weak_label'].isna().sum())
            }
        
        # 统计后验概率（如果存在）
        if 'weak_label_posterior_prob' in df.columns:
            posterior = df['weak_label_posterior_prob'].dropna()
            if len(posterior) > 0:
                stats['posterior_prob'] = {
                    'mean': float(posterior.mean()),
                    'median': float(posterior.median()),
                    'max': float(posterior.max()),
                    'samples_with_prob': len(posterior)
                }
        
        return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='BayesReviewNet - 基于贝叶斯网络的虚假评论识别（多视角特征）'
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
        choices=['amazon', 'yelp', 'all'],
        default=['all'],
        help='要处理的数据集 (amazon, yelp)'
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
        datasets = ['amazon', 'yelp']
    else:
        datasets = args.datasets
    
    # 初始化Pipeline
    pipeline = BayesReviewNetPipeline(args.config)
    
    # 处理每个数据集并收集统计信息
    results = {}
    for dataset_name in datasets:
        try:
            stats = pipeline.run(dataset_name, args.structure)
            results[dataset_name] = stats
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 时出错: {e}", exc_info=True)
            results[dataset_name] = {'status': 'failed', 'error': str(e)}
            continue
    
    # 打印汇总统计
    _print_summary(results)
    
    logger.info("\n🎉 所有任务完成！")


def _print_summary(results: dict):
    """
    打印处理结果汇总
    
    Args:
        results: 各数据集的处理结果字典
    """
    logger.info("\n" + "="*80)
    logger.info("处理结果汇总")
    logger.info("="*80)
    
    for dataset_name, stats in results.items():
        if stats.get('status') == 'failed':
            logger.info(f"\n❌ {dataset_name.upper()}: 处理失败")
            logger.info(f"   错误: {stats.get('error', 'Unknown')}")
            continue
        
        logger.info(f"\n✅ {dataset_name.upper()}")
        logger.info(f"   样本数: {stats['total_samples']:,}")
        logger.info(f"   特征统计:")
        logger.info(f"      - Text特征: {stats['features']['text']} 个")
        logger.info(f"      - Behavior特征: {stats['features']['behavior']} 个")
        logger.info(f"      - 离散化特征: {stats['features']['discrete']} 个")
        
        # 弱标签分布
        if 'weak_label_distribution' in stats:
            dist = stats['weak_label_distribution']
            total_labeled = dist['suspicious'] + dist['normal']
            if total_labeled > 0:
                susp_rate = dist['suspicious'] / total_labeled * 100
                logger.info(f"   弱标签分布:")
                logger.info(f"      - 可疑: {dist['suspicious']:,} ({susp_rate:.1f}%)")
                logger.info(f"      - 正常: {dist['normal']:,} ({100-susp_rate:.1f}%)")
                if dist['missing'] > 0:
                    logger.info(f"      - 缺失: {dist['missing']:,}")
        
        # 后验概率统计
        if 'posterior_prob' in stats:
            post = stats['posterior_prob']
            logger.info(f"   后验概率:")
            logger.info(f"      - 平均: {post['mean']:.4f}")
            logger.info(f"      - 中位数: {post['median']:.4f}")
            logger.info(f"      - 最大值: {post['max']:.4f}")
            logger.info(f"      - 有效样本: {post['samples_with_prob']:,}")
        
        logger.info(f"   输出文件: {stats['output_file']}")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
