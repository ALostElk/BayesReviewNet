#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yelp-Only 对照实验
=================

实验目标：
1. 完全基于 Yelp Open Dataset 训练和评估 BayesReviewNet
2. 消除跨域干扰，评估模型在单域内的性能上限
3. 为 Amazon→Yelp 跨域迁移提供对照基线

数据划分：
- Training Set (70%): 用于学习先验概率和条件概率(CPD)
- Validation Set (15%): 用于超参数调优和阈值选择
- Test Set (15%): 严格的最终评估

弱标签构造：
- 基于平台标注（filtered reviews）
- 行为异常（burstiness, rating variance）
- 文本异常（长度、重复度、情绪）
- 网络异常（reviewer-business 图结构）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import json

from src.utils.logging import setup_logger
from src.utils.config import load_config, ensure_dir
from src.utils.io import save_data
from src.preprocessing import YelpPreprocessor
from src.preprocessing.weak_labeling import construct_weak_label
from src.features import TextFeatureExtractor, BehaviorFeatureExtractor, FeatureDiscretizer
from src.bayes import BayesianNetworkStructure, CPDLearner, BayesianInference
from src.evaluation import evaluate_model
from src.evaluation.metrics import find_optimal_threshold

logger = setup_logger("yelp_only_exp")


class YelpOnlyExperiment:
    """
    Yelp-Only 对照实验
    
    完全基于 Yelp 数据训练和评估，不使用任何 Amazon 数据
    """
    
    def __init__(self, config_path: str = 'configs/default.yaml', use_sampling: bool = True):
        """
        初始化实验
        
        Args:
            config_path: 配置文件路径
            use_sampling: 是否使用配置文件中的采样设置
        """
        self.config = load_config(config_path)
        self.use_sampling = use_sampling
        self.results = {}
        
        logger.info("="*80)
        logger.info("Yelp-Only 对照实验")
        logger.info("="*80)
        logger.info("实验目标: 评估 BayesReviewNet 在单域内的性能上限")
        logger.info("数据来源: Yelp Open Dataset (仅)")
        logger.info("="*80)
    
    def run(self, 
            sample_size: int = None,
            train_ratio: float = 0.70,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15,
            random_seed: int = 42) -> Dict:
        """
        运行完整的 Yelp-Only 实验
        
        Args:
            sample_size: 采样大小（None表示全量）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            
        Returns:
            实验结果字典
        """
        np.random.seed(random_seed)
        
        # ========== 步骤1: 数据加载与预处理 ==========
        logger.info("\n【步骤1】加载 Yelp 数据")
        yelp_df = self._load_yelp_data(sample_size)
        
        # ========== 步骤2: 特征工程 ==========
        logger.info("\n【步骤2】特征工程")
        yelp_df = self._extract_features(yelp_df)
        
        # ========== 步骤3: 构造弱标签 ==========
        logger.info("\n【步骤3】构造弱监督标签")
        yelp_df = self._construct_weak_labels(yelp_df)
        
        # ========== 步骤4: 数据集划分 ==========
        logger.info("\n【步骤4】数据集划分 (Train/Val/Test)")
        train_df, val_df, test_df = self._split_dataset(
            yelp_df, train_ratio, val_ratio, test_ratio, random_seed
        )
        
        # ========== 步骤5: 训练贝叶斯网络 ==========
        logger.info("\n【步骤5】训练贝叶斯网络 (仅基于 Yelp Training Set)")
        structure, cpd_learner = self._train_bayesian_network(train_df)
        
        # ========== 步骤6: 验证集调优 ==========
        logger.info("\n【步骤6】验证集性能评估与阈值选择")
        val_results, optimal_threshold = self._evaluate_on_validation(
            val_df, structure, cpd_learner
        )
        
        # ========== 步骤7: 测试集评估 ==========
        logger.info("\n【步骤7】测试集最终评估")
        test_results = self._evaluate_on_test(
            test_df, structure, cpd_learner, optimal_threshold
        )
        
        # ========== 步骤8: 保存结果 ==========
        logger.info("\n【步骤8】保存实验结果")
        self._save_results(train_df, val_df, test_df, val_results, test_results)
        
        # ========== 步骤9: 生成对比报告 ==========
        logger.info("\n【步骤9】生成对比分析报告")
        self._generate_comparison_report(val_results, test_results)
        
        return {
            'validation': val_results,
            'test': test_results,
            'optimal_threshold': optimal_threshold,
            'data_split': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df)
            }
        }
    
    def _load_yelp_data(self, sample_size: int = None) -> pd.DataFrame:
        """
        加载 Yelp 原始数据
        
        Args:
            sample_size: 采样大小（优先级高于配置文件）
            
        Returns:
            标准化的 DataFrame
        """
        yelp_preprocessor = YelpPreprocessor(
            self.config['data_paths']['yelp']['raw_dir']
        )
        
        # 确定最终采样大小
        if sample_size is not None:
            # 命令行参数优先
            final_sample_size = sample_size
            logger.info(f"  使用命令行指定的采样大小: {final_sample_size}")
        elif self.use_sampling and self.config.get('sampling', {}).get('enabled', False):
            # 使用配置文件
            final_sample_size = self.config['sampling'].get('yelp_sample_size', None)
            logger.info(f"  使用配置文件的采样大小: {final_sample_size}")
        else:
            # 全量数据
            final_sample_size = None
            logger.info(f"  使用全量数据（无采样）")
        
        df = yelp_preprocessor.load_and_standardize(final_sample_size)
        
        logger.info(f"  ✓ 加载 Yelp 数据: {len(df)} 条记录")
        logger.info(f"  - 时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        logger.info(f"  - 用户数: {df['user_id'].nunique()}")
        logger.info(f"  - 商家数: {df['item_id'].nunique()}")
        
        return df
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取多视角特征
        
        Args:
            df: 原始 DataFrame
            
        Returns:
            添加特征后的 DataFrame
        """
        # 文本特征
        logger.info("  → 提取文本特征...")
        text_extractor = TextFeatureExtractor()
        df = text_extractor.extract(df)
        
        # 行为特征
        logger.info("  → 提取行为特征...")
        behavior_extractor = BehaviorFeatureExtractor()
        df = behavior_extractor.extract(df)
        
        # 特征离散化
        logger.info("  → 特征离散化...")
        discretizer = FeatureDiscretizer()
        df = discretizer.discretize(df)
        
        # 特征质量检查
        self._check_feature_quality(df)
        
        return df
    
    def _check_feature_quality(self, df: pd.DataFrame):
        """
        检查特征质量，发出警告
        
        Args:
            df: DataFrame
        """
        discrete_cols = [c for c in df.columns if c.endswith('_discrete')]
        
        logger.info("  → 特征质量检查:")
        
        warnings = []
        
        for col in discrete_cols:
            # 检查唯一值数量
            n_unique = df[col].nunique()
            missing_rate = df[col].isna().mean()
            
            if n_unique == 1:
                warnings.append(f"    ⚠ {col}: 无区分度（仅1个唯一值）")
            elif missing_rate > 0.5:
                warnings.append(f"    ⚠ {col}: 高缺失率 ({missing_rate*100:.1f}%)")
        
        if warnings:
            logger.warning("  发现特征质量问题:")
            for w in warnings:
                logger.warning(w)
        else:
            logger.info("    ✓ 所有特征质量正常")
    
    def _construct_weak_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构造 Yelp 弱监督标签
        
        基于多个信号：
        1. 平台标注 (filtered reviews)
        2. 行为异常
        3. 文本异常
        4. 网络异常
        
        Args:
            df: DataFrame
            
        Returns:
            添加 weak_label 的 DataFrame
        """
        df = construct_weak_label(df, 'yelp')
        
        # 统计弱标签分布
        label_dist = df['weak_label'].value_counts()
        total = len(df)
        
        logger.info(f"  ✓ 弱标签构造完成:")
        logger.info(f"    - Fraud (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/total*100:.1f}%)")
        logger.info(f"    - Normal (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/total*100:.1f}%)")
        logger.info(f"    - Missing: {df['weak_label'].isna().sum()}")
        
        return df
    
    def _split_dataset(self,
                       df: pd.DataFrame,
                       train_ratio: float,
                       val_ratio: float,
                       test_ratio: float,
                       random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分数据集为 Train / Validation / Test
        
        使用分层采样确保标签分布一致
        
        Args:
            df: 完整 DataFrame
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            
        Returns:
            (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # 验证比例
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        # 移除 weak_label 缺失的样本
        df_valid = df[df['weak_label'].notna()].copy()
        logger.info(f"  有效样本（weak_label非空）: {len(df_valid)} / {len(df)}")
        
        # 第一次划分: train vs (val+test)
        train_df, temp_df = train_test_split(
            df_valid,
            train_size=train_ratio,
            stratify=df_valid['weak_label'],
            random_state=random_seed
        )
        
        # 第二次划分: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            stratify=temp_df['weak_label'],
            random_state=random_seed
        )
        
        logger.info(f"  ✓ 数据集划分完成:")
        logger.info(f"    - Training:   {len(train_df)} ({len(train_df)/len(df_valid)*100:.1f}%)")
        logger.info(f"    - Validation: {len(val_df)} ({len(val_df)/len(df_valid)*100:.1f}%)")
        logger.info(f"    - Test:       {len(test_df)} ({len(test_df)/len(df_valid)*100:.1f}%)")
        
        # 验证标签分布
        logger.info(f"\n  标签分布一致性检查:")
        for name, subset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            fraud_rate = subset['weak_label'].mean()
            logger.info(f"    {name}: Fraud率 = {fraud_rate*100:.1f}%")
        
        return train_df, val_df, test_df
    
    def _train_bayesian_network(self,
                                 train_df: pd.DataFrame) -> Tuple[BayesianNetworkStructure, CPDLearner]:
        """
        训练贝叶斯网络
        
        完全基于 Yelp Training Set，不使用任何外部数据
        
        Args:
            train_df: 训练数据
            
        Returns:
            (structure, cpd_learner)
        """
        # 定义网络结构
        structure = BayesianNetworkStructure()
        structure.define_structure('default')
        
        logger.info(f"  ✓ DAG 结构: {len(structure.edges)} 条边")
        logger.info(f"  拓扑排序: {structure.get_topological_order()}")
        
        # 学习条件概率分布
        cpd_learner = CPDLearner(structure)
        cpd_learner.learn_cpds(train_df, smoothing=1.0)
        
        logger.info(f"  ✓ CPD 学习完成")
        logger.info(f"    - 训练样本: {len(train_df)}")
        logger.info(f"    - 学习节点: {len(cpd_learner.cpds)}")
        
        return structure, cpd_learner
    
    def _evaluate_on_validation(self,
                                 val_df: pd.DataFrame,
                                 structure: BayesianNetworkStructure,
                                 cpd_learner: CPDLearner) -> Tuple[Dict, float]:
        """
        在验证集上评估并选择最优阈值
        
        Args:
            val_df: 验证数据
            structure: 网络结构
            cpd_learner: CPD学习器
            
        Returns:
            (evaluation_results, optimal_threshold)
        """
        # 推断
        inference = BayesianInference(structure, cpd_learner)
        val_df = inference.infer_posterior(val_df.copy(), target_variable='weak_label')
        
        # 后验概率统计
        posterior = val_df['weak_label_posterior_prob'].dropna()
        logger.info(f"  后验概率分布:")
        logger.info(f"    - 均值: {posterior.mean():.4f}")
        logger.info(f"    - 中位数: {posterior.median():.4f}")
        logger.info(f"    - 标准差: {posterior.std():.4f}")
        logger.info(f"    - 最小值: {posterior.min():.4f}")
        logger.info(f"    - 最大值: {posterior.max():.4f}")
        
        # 寻找最优阈值
        logger.info(f"\n  寻找最优分类阈值 (基于 F1-Score)...")
        optimal_result = find_optimal_threshold(val_df, metric='f1')
        optimal_threshold = optimal_result['best_threshold']
        
        logger.info(f"  ✓ 最优阈值: {optimal_threshold:.4f}")
        
        # 使用最优阈值评估
        results = evaluate_model(val_df, threshold=optimal_threshold)
        metrics = results['metrics']
        
        logger.info(f"\n  验证集性能 (阈值={optimal_threshold:.4f}):")
        logger.info(f"    - Precision: {metrics['precision']:.4f}")
        logger.info(f"    - Recall:    {metrics['recall']:.4f}")
        logger.info(f"    - F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"    - ROC-AUC:   {metrics.get('roc_auc', 'N/A')}")
        
        return results, optimal_threshold
    
    def _evaluate_on_test(self,
                          test_df: pd.DataFrame,
                          structure: BayesianNetworkStructure,
                          cpd_learner: CPDLearner,
                          threshold: float) -> Dict:
        """
        在测试集上最终评估
        
        Args:
            test_df: 测试数据
            structure: 网络结构
            cpd_learner: CPD学习器
            threshold: 分类阈值
            
        Returns:
            evaluation_results
        """
        # 推断
        inference = BayesianInference(structure, cpd_learner)
        test_df = inference.infer_posterior(test_df.copy(), target_variable='weak_label')
        
        # 后验概率统计
        posterior = test_df['weak_label_posterior_prob'].dropna()
        logger.info(f"  后验概率分布:")
        logger.info(f"    - 均值: {posterior.mean():.4f}")
        logger.info(f"    - 中位数: {posterior.median():.4f}")
        logger.info(f"    - 标准差: {posterior.std():.4f}")
        
        # 评估
        results = evaluate_model(test_df, threshold=threshold)
        metrics = results['metrics']
        
        logger.info(f"\n  测试集最终性能 (阈值={threshold:.4f}):")
        logger.info(f"    - Precision: {metrics['precision']:.4f}")
        logger.info(f"    - Recall:    {metrics['recall']:.4f}")
        logger.info(f"    - F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"    - ROC-AUC:   {metrics.get('roc_auc', 'N/A')}")
        
        return results
    
    def _save_results(self,
                      train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      val_results: Dict,
                      test_results: Dict):
        """
        保存实验结果
        
        Args:
            train_df: 训练数据
            val_df: 验证数据
            test_df: 测试数据
            val_results: 验证集结果
            test_results: 测试集结果
        """
        output_dir = Path('data/experiments/yelp_only')
        ensure_dir(str(output_dir))
        
        # 保存数据
        save_data(train_df, str(output_dir / 'train.parquet'))
        save_data(val_df, str(output_dir / 'validation.parquet'))
        save_data(test_df, str(output_dir / 'test.parquet'))
        
        # 保存结果
        results_summary = {
            'experiment': 'Yelp-Only Baseline',
            'data_split': {
                'train': len(train_df),
                'validation': len(val_df),
                'test': len(test_df)
            },
            'validation_metrics': val_results['metrics'],
            'test_metrics': test_results['metrics']
        }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"  ✓ 结果已保存到: {output_dir}/")
    
    def _generate_comparison_report(self, val_results: Dict, test_results: Dict):
        """
        生成对比分析报告
        
        Args:
            val_results: 验证集结果
            test_results: 测试集结果
        """
        logger.info("\n" + "="*80)
        logger.info("Yelp-Only 实验总结")
        logger.info("="*80)
        
        logger.info("\n【模型性能】")
        logger.info("  验证集:")
        for metric, value in val_results['metrics'].items():
            logger.info(f"    {metric:12s}: {value:.4f}")
        
        logger.info("\n  测试集:")
        for metric, value in test_results['metrics'].items():
            logger.info(f"    {metric:12s}: {value:.4f}")
        
        logger.info("\n【关键观察】")
        
        # 泛化能力
        val_f1 = val_results['metrics']['f1']
        test_f1 = test_results['metrics']['f1']
        generalization_gap = val_f1 - test_f1
        
        logger.info(f"  泛化差距 (Val F1 - Test F1): {generalization_gap:+.4f}")
        if abs(generalization_gap) < 0.02:
            logger.info("    → 泛化能力良好")
        elif generalization_gap > 0:
            logger.info("    → 存在轻微过拟合")
        else:
            logger.info("    → 测试集性能优于验证集（正常波动）")
        
        logger.info("\n【对照实验意义】")
        logger.info("  此结果为 Amazon→Yelp 跨域迁移提供对照基线:")
        logger.info("  - 如果跨域迁移 F1 < Yelp-Only F1:")
        logger.info("    → 说明跨域干扰确实存在，需要域适应")
        logger.info("  - 如果跨域迁移 F1 ≈ Yelp-Only F1:")
        logger.info("    → 说明跨域迁移效果良好，已接近单域上限")
        logger.info("  - 如果跨域迁移 F1 > Yelp-Only F1:")
        logger.info("    → 说明 Amazon 数据带来了正迁移增益")
        
        logger.info("\n" + "="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Yelp-Only 对照实验 - 评估 BayesReviewNet 单域性能上限',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用配置文件中的采样设置（默认）
  python yelp_only_experiment.py
  
  # 指定采样大小
  python yelp_only_experiment.py --sample-size 10000
  
  # 使用全量数据（忽略配置文件）
  python yelp_only_experiment.py --no-sampling
  
  # 自定义数据划分
  python yelp_only_experiment.py --sample-size 50000 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        """
    )
    
    # 数据采样选项
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='采样大小（指定后优先级高于配置文件，None表示使用配置文件设置）'
    )
    sampling_group.add_argument(
        '--no-sampling',
        action='store_true',
        help='不使用采样，加载全量数据（会覆盖配置文件设置）'
    )
    
    # 数据划分选项
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='训练集比例（默认0.70）'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='验证集比例（默认0.15）'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='测试集比例（默认0.15）'
    )
    
    # 其他选项
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径（默认configs/default.yaml）'
    )
    
    args = parser.parse_args()
    
    # 确定采样策略
    if args.no_sampling:
        use_sampling = False
        sample_size = None
        logger.info("模式: 全量数据（不采样）")
    elif args.sample_size is not None:
        use_sampling = True
        sample_size = args.sample_size
        logger.info(f"模式: 指定采样大小 = {sample_size}")
    else:
        use_sampling = True
        sample_size = None
        logger.info("模式: 使用配置文件的采样设置")
    
    # 运行实验
    experiment = YelpOnlyExperiment(
        config_path=args.config,
        use_sampling=use_sampling
    )
    results = experiment.run(
        sample_size=sample_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    logger.info("\n🎉 Yelp-Only 对照实验完成！")


if __name__ == '__main__':
    main()

