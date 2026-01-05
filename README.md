# 基于贝叶斯网络的虚假评论识别框架

## 项目简介

为贝叶斯网络构建可解释的特征变量，用于虚假评论识别研究。

**核心理念**：评论欺诈由用户行为、时间模式、平台机制与文本特征共同驱动。

**技术特点**：
- ✅ 无深度学习，仅用统计和词典方法
- ✅ 所有特征可解释、可复现
- ✅ 变量离散化为有限状态

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 测试环境
python test_installation.py

# 3. 快速测试（采样模式）
python main.py --datasets all --sample

# 4. 查看结果
python quick_start.py
```

## 数据集

详见 [DATASET.md](DATASET.md)

- **Amazon Reviews**: 训练集，弱标签
- **OpSpam**: 测试集，真实标签（仅测试用）
- **Yelp**: 测试集，弱标签

## 特征体系

### 文本特征（10个）
- review_length, sentiment_score, subjectivity_score
- exclamation_ratio, first_person_pronoun_ratio
- capital_letter_ratio, word_count, sentence_count
- avg_word_length, unique_word_ratio

### 用户特征（8个）
- user_review_count, user_avg_rating, user_rating_std
- user_rating_deviation, user_rating_entropy
- user_verified_ratio, user_review_burstiness
- user_avg_review_length

### 离散化特征（7个）
关键特征自动离散化为有限状态（SHORT/NORMAL/LONG等）

## 弱标签规则

基于启发式规则构造（阈值≥4分标记为可疑）：

**Amazon**: 未验证购买+2, 评分极端+1, 长度异常+1, 爆发性高+2, 评分偏差+1, 评论过多+1

**Yelp**: 评分极端+1, 长度异常+1, 爆发性高+2, 评分熵低+2, 代词异常+1, 情感极端+1

**OpSpam**: 真实标签（仅测试用）

## 输出文件

- `discretized/`: 最终数据（Parquet + CSV）
- `metadata/`: 特征描述、规则、统计
- `logs/`: 处理日志

## 使用建议

1. **首次运行**: 使用 `--sample` 快速测试
2. **OpSpam仅测试**: 不参与训练
3. **弱标签有噪声**: Amazon和Yelp标签是启发式的
4. **可调整规则**: 编辑 `config.yaml` 自定义

## 技术栈

Python 3.8+ | pandas | numpy | scipy | textblob | pyyaml

