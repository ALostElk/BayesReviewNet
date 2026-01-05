#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安装测试脚本
验证所有依赖是否正确安装
"""
import sys

def test_imports():
    """测试所有必需的包是否可以导入"""
    print("测试依赖包导入...")
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'textblob': 'textblob',
        'yaml': 'pyyaml',
        'tqdm': 'tqdm'
    }
    
    failed = []
    
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError as e:
            print(f"  ✗ {package_name}: {e}")
            failed.append(package_name)
    
    return failed


def test_project_structure():
    """测试项目结构是否完整"""
    print("\n测试项目结构...")
    
    import os
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'main.py',
        'quick_start.py',
        'README.md',
        'USAGE.md',
        'src/__init__.py',
        'src/utils.py',
        'src/data_loader.py',
        'src/text_features.py',
        'src/user_features.py',
        'src/discretizer.py',
        'src/weak_labeler.py'
    ]
    
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}")
            missing.append(file_path)
    
    return missing


def test_config():
    """测试配置文件是否可以加载"""
    print("\n测试配置文件...")
    
    try:
        from src.utils import load_config
        config = load_config('config.yaml')
        print(f"  ✓ 配置文件加载成功")
        print(f"  ✓ 包含 {len(config)} 个配置项")
        return True
    except Exception as e:
        print(f"  ✗ 配置文件加载失败: {e}")
        return False


def test_data_directories():
    """测试数据目录是否存在"""
    print("\n测试数据目录...")
    
    import os
    
    data_dirs = [
        '数据集/训练集/Amazon Review Data',
        '数据集/训练集/op_spam_AI-main',
        '数据集/测试集/Yelp-JSON'
    ]
    
    missing = []
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ⚠ {dir_path} (未找到，运行时需要)")
            missing.append(dir_path)
    
    return missing


def test_textblob():
    """测试TextBlob是否可用"""
    print("\n测试TextBlob...")
    
    try:
        from textblob import TextBlob
        
        # 测试基本功能
        text = "This is a great product!"
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        print(f"  ✓ TextBlob可用")
        print(f"  ✓ 情感分析功能正常 (polarity={sentiment.polarity:.2f})")
        return True
    except Exception as e:
        print(f"  ✗ TextBlob测试失败: {e}")
        print(f"  提示: 可能需要下载NLTK数据")
        print(f"  运行: python -c \"import nltk; nltk.download('punkt'); nltk.download('brown')\"")
        return False


def main():
    """主测试函数"""
    print("="*80)
    print("贝叶斯网络虚假评论识别框架 - 安装测试")
    print("="*80)
    
    all_passed = True
    
    # 测试依赖包
    failed_imports = test_imports()
    if failed_imports:
        print(f"\n⚠ 缺少以下依赖包: {', '.join(failed_imports)}")
        print(f"  请运行: pip install {' '.join(failed_imports)}")
        all_passed = False
    
    # 测试项目结构
    missing_files = test_project_structure()
    if missing_files:
        print(f"\n⚠ 缺少以下文件: {', '.join(missing_files)}")
        all_passed = False
    
    # 测试配置
    if not test_config():
        all_passed = False
    
    # 测试数据目录
    missing_dirs = test_data_directories()
    if missing_dirs:
        print(f"\n⚠ 提示: 运行前需要准备数据集")
    
    # 测试TextBlob
    if not test_textblob():
        all_passed = False
    
    # 总结
    print("\n" + "="*80)
    if all_passed:
        print("✅ 所有测试通过！环境配置正确。")
        print("\n下一步:")
        print("  1. 准备数据集（如果还没有）")
        print("  2. 运行: python main.py --datasets all --sample")
        print("  3. 查看结果: python quick_start.py")
    else:
        print("❌ 部分测试失败，请根据上述提示修复。")
        print("\n常见问题:")
        print("  1. 依赖包缺失: pip install -r requirements.txt")
        print("  2. NLTK数据缺失: python -c \"import nltk; nltk.download('punkt')\"")
        print("  3. 文件缺失: 检查项目完整性")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

