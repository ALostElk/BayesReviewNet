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
        
        # 2.4 特征离散化（数据驱动的分位数离散化）
        logger.info("  → 特征离散化（基于分位数）...")
        discretizer = FeatureDiscretizer()  # 不再需要config参数
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
    parser.add_argument(
        '--cross-domain',
        action='store_true',
        help='启用跨域迁移模式（Amazon→Yelp，使用似然校准）'
    )
    parser.add_argument(
        '--use-validation',
        action='store_true',
        default=True,
        help='使用预定义的验证集进行校准（默认True）'
    )
    parser.add_argument(
        '--calibration-ratio',
        type=float,
        default=0.20,
        help='校准集比例（仅在--no-use-validation时生效，默认0.20）'
    )
    parser.add_argument(
        '--calibration-strength',
        type=float,
        default=0.3,
        help='校准强度α（默认0.3，范围0-1）'
    )
    
    args = parser.parse_args()
    
    # 如果启用跨域迁移模式，使用专门的pipeline
    if args.cross_domain:
        logger.info("\n" + "="*80)
        logger.info("跨域迁移模式 (Cross-Domain Transfer Mode)")
        logger.info("="*80)
        _run_cross_domain_transfer(args)
        return
    
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


def _run_cross_domain_transfer(args):
    """
    运行跨域迁移学习
    
    Amazon (源域) → Yelp (目标域) + 似然校准
    
    Args:
        args: 命令行参数
    """
    from utils.config import load_config
    from utils.data_split import split_for_calibration, validate_split
    from preprocessing import AmazonPreprocessor, YelpPreprocessor
    from preprocessing.weak_labeling import construct_weak_label
    from features import TextFeatureExtractor, BehaviorFeatureExtractor, FeatureDiscretizer
    from bayes import BayesianNetworkStructure, CPDLearner, BayesianInference, LikelihoodCalibrator
    from evaluation import evaluate_model
    
    config = load_config(args.config)
    
    logger.info(f"配置:")
    logger.info(f"  - 使用验证集: {args.use_validation}")
    if not args.use_validation:
        logger.info(f"  - 校准集比例: {args.calibration_ratio*100:.0f}%")
    logger.info(f"  - 校准强度 α: {args.calibration_strength}")
    logger.info(f"  - 网络结构: {args.structure}")
    
    # ========== 步骤1: 准备Amazon源域数据 ==========
    logger.info("\n【步骤1】准备Amazon源域数据")
    amazon_preprocessor = AmazonPreprocessor(config['data_paths']['amazon']['raw_dir'])
    amazon_sample_size = config['sampling']['amazon_sample_size'] \
        if config['sampling']['enabled'] else None
    amazon_df = amazon_preprocessor.load_and_standardize(amazon_sample_size)
    
    # 特征提取
    logger.info("  → 提取特征...")
    amazon_df = _extract_all_features(amazon_df)
    amazon_df = construct_weak_label(amazon_df, 'amazon')
    logger.info(f"  ✓ Amazon数据: {len(amazon_df)} 条")
    
    # ========== 步骤2: 在Amazon上训练贝叶斯网络 ==========
    logger.info("\n【步骤2】在Amazon上训练贝叶斯网络（源域）")
    structure = BayesianNetworkStructure()
    structure.define_structure(args.structure)
    logger.info(f"  ✓ DAG结构: {len(structure.edges)} 条边")
    
    amazon_cpd = CPDLearner(structure)
    amazon_cpd.learn_cpds(amazon_df, smoothing=1.0)
    logger.info(f"  ✓ CPD学习完成（源域知识）")
    
    # ========== 步骤3: 准备Yelp目标域数据并划分 ==========
    logger.info("\n【步骤3】准备Yelp目标域数据并划分")
    yelp_preprocessor = YelpPreprocessor(config['data_paths']['yelp']['raw_dir'])
    
    if args.use_validation:
        # 使用固定的验证集划分（首次运行时创建，后续重用）
        logger.info("  → 使用固定的验证集划分（确保可重复性）")
        
        yelp_calib, yelp_test = _load_or_create_fixed_split(
            yelp_preprocessor,
            config,
            calibration_ratio=args.calibration_ratio
        )
        
        logger.info(f"  ✓ 验证集（校准用）: {len(yelp_calib)} 条")
        logger.info(f"  ✓ 测试集: {len(yelp_test)} 条")
        
    else:
        # 随机划分方式（每次运行重新划分）
        yelp_sample_size = config['sampling']['yelp_sample_size'] \
            if config['sampling']['enabled'] else None
        yelp_df = yelp_preprocessor.load_and_standardize(yelp_sample_size)
        
        # 特征提取
        logger.info("  → 提取特征...")
        yelp_df = _extract_all_features(yelp_df)
        yelp_df = construct_weak_label(yelp_df, 'yelp')
        logger.info(f"  ✓ Yelp数据: {len(yelp_df)} 条")
        
        # 划分为校准集和测试集
        logger.info(f"  → 随机划分数据（校准:{args.calibration_ratio*100:.0f}% / 测试:{(1-args.calibration_ratio)*100:.0f}%）")
        yelp_calib, yelp_test = split_for_calibration(
            yelp_df,
            calibration_ratio=args.calibration_ratio,
            stratify_by='weak_label',
            random_state=42
        )
        validate_split(yelp_calib, yelp_test, label_col='weak_label')
    
    # ========== 步骤4: 执行似然校准 ==========
    logger.info(f"\n【步骤4】执行似然校准（α={args.calibration_strength}）")
    calibrator = LikelihoodCalibrator(
        amazon_cpd,
        calibration_strength=args.calibration_strength
    )
    calibrator.calibrate(yelp_calib, target_variable='weak_label')
    
    calibrated_cpd = calibrator.get_calibrated_cpd_learner()
    calib_report = calibrator.get_calibration_report()
    logger.info(f"  ✓ 校准完成:")
    logger.info(f"    - 总节点: {calib_report['total_nodes']}")
    logger.info(f"    - 已校准: {calib_report['calibrated_nodes']}")
    logger.info(f"    - 保持不变: {calib_report['kept_nodes']}")
    
    # ========== 步骤5: 在测试集上评估 ==========
    logger.info("\n【步骤5】在Yelp测试集上评估")
    
    # 5a. 基线（无校准）
    logger.info("  → 基线性能（无校准）:")
    baseline_results = _evaluate_on_test(yelp_test, structure, amazon_cpd)
    
    # 5b. 校准后
    logger.info("\n  → 校准后性能:")
    calibrated_results = _evaluate_on_test(yelp_test, structure, calibrated_cpd)
    
    # ========== 步骤6: 对比分析 ==========
    logger.info("\n【步骤6】性能对比分析")
    _compare_performance(baseline_results, calibrated_results)
    
    # ========== 步骤7: 保存结果 ==========
    logger.info("\n【步骤7】保存结果")
    output_path = f"data/processed/yelp_calibrated_r{int(args.calibration_ratio*100)}_a{int(args.calibration_strength*100)}.parquet"
    
    # 在校准后的CPD上进行推断
    inference = BayesianInference(structure, calibrated_cpd)
    yelp_test = inference.infer_posterior(yelp_test, target_variable='weak_label')
    
    from utils.io import save_data
    save_data(yelp_test, output_path)
    logger.info(f"  ✓ 结果已保存: {output_path}")
    
    logger.info("\n" + "="*80)
    logger.info("🎉 跨域迁移学习完成！")
    logger.info("="*80)


def _load_or_create_fixed_split(yelp_preprocessor, config, calibration_ratio=0.20):
    """
    加载或创建固定的Yelp验证集/测试集划分
    
    首次运行时创建划分并保存索引，后续运行重用相同的划分
    这确保了跨域实验的可重复性
    
    Args:
        yelp_preprocessor: Yelp预处理器
        config: 配置字典
        calibration_ratio: 验证集比例
        
    Returns:
        (validation_df, test_df)
    """
    from pathlib import Path
    import pickle
    from preprocessing.weak_labeling import construct_weak_label
    
    # 划分索引文件路径
    split_file = Path('data/processed/yelp_fixed_split_indices.pkl')
    
    # 加载完整的Yelp数据
    yelp_sample_size = config['sampling']['yelp_sample_size'] \
        if config['sampling']['enabled'] else None
    yelp_df = yelp_preprocessor.load_and_standardize(yelp_sample_size)
    
    # 特征提取（在划分之前）
    logger.info("  → 提取特征...")
    yelp_df = _extract_all_features(yelp_df)
    yelp_df = construct_weak_label(yelp_df, 'yelp')
    
    # 检查是否存在固定划分
    if split_file.exists():
        logger.info(f"  → 加载固定划分索引: {split_file}")
        with open(split_file, 'rb') as f:
            split_indices = pickle.load(f)
        
        val_indices = split_indices['validation']
        test_indices = split_indices['test']
        
        # 使用保存的索引划分数据
        validation_df = yelp_df.iloc[val_indices].copy()
        test_df = yelp_df.iloc[test_indices].copy()
        
        logger.info(f"  ✓ 使用固定划分（验证集:{len(validation_df)}, 测试集:{len(test_df)}）")
        
    else:
        logger.info(f"  → 创建新的固定划分（验证集:{calibration_ratio*100:.0f}%）")
        
        # 创建新划分
        from utils.data_split import split_for_calibration, validate_split
        
        validation_df, test_df = split_for_calibration(
            yelp_df,
            calibration_ratio=calibration_ratio,
            stratify_by='weak_label',
            random_state=42
        )
        
        # 保存索引以供后续使用
        split_indices = {
            'validation': validation_df.index.tolist(),
            'test': test_df.index.tolist(),
            'calibration_ratio': calibration_ratio,
            'total_samples': len(yelp_df)
        }
        
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, 'wb') as f:
            pickle.dump(split_indices, f)
        
        logger.info(f"  ✓ 固定划分已保存: {split_file}")
        
        # 验证划分
        validate_split(validation_df, test_df, label_col='weak_label')
    
    return validation_df, test_df


def _extract_all_features(df):
    """辅助函数：提取所有特征"""
    text_extractor = TextFeatureExtractor()
    df = text_extractor.extract(df)
    
    behavior_extractor = BehaviorFeatureExtractor()
    df = behavior_extractor.extract(df)
    
    discretizer = FeatureDiscretizer()
    df = discretizer.discretize(df)
    
    return df


def _evaluate_on_test(test_df, structure, cpd_learner):
    """辅助函数：在测试集上评估"""
    from evaluation.metrics import find_optimal_threshold
    
    inference = BayesianInference(structure, cpd_learner)
    test_df = inference.infer_posterior(test_df.copy(), target_variable='weak_label')
    
    # 后验概率统计
    if 'weak_label_posterior_prob' in test_df.columns:
        posterior = test_df['weak_label_posterior_prob'].dropna()
        if len(posterior) > 0:
            logger.info(f"    后验均值: {posterior.mean():.4f}")
            logger.info(f"    后验中位数: {posterior.median():.4f}")
    
    # 找到最优阈值
    optimal_result = find_optimal_threshold(test_df, metric='f1')
    optimal_threshold = optimal_result['best_threshold']
    logger.info(f"    最优阈值: {optimal_threshold:.4f}")
    
    # 使用最优阈值评估
    results = evaluate_model(test_df, threshold=optimal_threshold)
    metrics = results['metrics']
    
    logger.info(f"    Precision: {metrics['precision']:.4f}")
    logger.info(f"    Recall:    {metrics['recall']:.4f}")
    logger.info(f"    F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"    ROC-AUC:   {metrics.get('roc_auc', 'N/A')}")
    
    return results


def _compare_performance(baseline, calibrated):
    """辅助函数：对比性能"""
    logger.info("\n  ┌────────────┬──────────┬────────────┬─────────┐")
    logger.info("  │   指标     │ Baseline │ Calibrated │  提升   │")
    logger.info("  ├────────────┼──────────┼────────────┼─────────┤")
    
    metrics = ['precision', 'recall', 'f1', 'roc_auc']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for metric, name in zip(metrics, metric_names):
        base_val = baseline['metrics'].get(metric, 0.0)
        calib_val = calibrated['metrics'].get(metric, 0.0)
        improvement = calib_val - base_val
        
        improvement_str = f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"
        
        logger.info(
            f"  │ {name:10s} │ {base_val:8.4f} │  {calib_val:8.4f}  │ {improvement_str:7s} │"
        )
    
    logger.info("  └────────────┴──────────┴────────────┴─────────┘")
    
    # 总结
    f1_improvement = calibrated['metrics']['f1'] - baseline['metrics']['f1']
    if f1_improvement > 0.01:
        logger.info(f"\n  ✓ 校准有效！F1-Score提升 {f1_improvement:.4f}")
    elif f1_improvement > 0:
        logger.info(f"\n  → 校准略有改善，F1-Score提升 {f1_improvement:.4f}")
    else:
        logger.info(f"\n  ⚠ 校准未带来显著提升，F1-Score变化 {f1_improvement:.4f}")


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
