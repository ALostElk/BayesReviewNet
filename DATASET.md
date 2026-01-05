# 数据集说明

## 数据集概览

本项目使用三个数据集，分别用于训练和测试：

| 数据集 | 用途 | 标签类型 | 位置 |
|--------|------|----------|------|
| Amazon Reviews | 训练 | 弱标签（启发式） | `数据集/训练集/Amazon Review Data/` |
| OpSpam | 测试 | 真实标签 | `数据集/训练集/op_spam_AI-main/` |
| Yelp | 测试 | 弱标签（启发式） | `数据集/测试集/Yelp-JSON/` |

## 1. Amazon Reviews 5-core

### 数据来源
Amazon产品评论数据集，每个用户和产品至少有5条评论。

### 文件格式
- 格式：JSON Lines（每行一个JSON对象）
- 编码：UTF-8
- 文件示例：`Electronics_5.json`, `Books_5.json` 等

### 关键字段
```json
{
  "reviewerID": "用户ID",
  "asin": "商品ID", 
  "reviewText": "评论文本",
  "overall": 5.0,
  "verified": true,
  "vote": "10",
  "unixReviewTime": 1234567890
}
```

### 弱标签规则
基于以下6条规则构造（阈值≥4分标记为可疑）：
- 未验证购买 +2分
- 评分极端（1星或5星） +1分
- 评论长度异常 +1分
- 评论爆发性高 +2分
- 评分偏差大 +1分
- 用户评论数过多 +1分

## 2. OpSpam (AI生成评论)

### 数据来源
人工智能生成的虚假评论数据集，包含GPT-3和LLaMA生成的正面/负面评论。

### 文件结构
```
op_spam_AI-main/
├── generated_GPT3_positive/    # 400个正面评论
├── generated_GPT3_negative/    # 400个负面评论
├── generated_LLama_positive/   # 400个正面评论
└── generated_LLama_negative/   # 400个负面评论
```

### 文件格式
- 格式：纯文本（.txt）
- 每个文件一条评论
- 总计：1600条评论

### 标签
- **全部为虚假评论**（label=1）
- 提供真实标签（ground truth）
- **仅用于测试，不参与训练**

## 3. Yelp Open Dataset

### 数据来源
Yelp公开的商家评论数据集。

### 文件格式
- 格式：JSON Lines
- 编码：UTF-8
- 主要文件：
  - `yelp_academic_dataset_review.json` - 评论数据
  - `yelp_academic_dataset_user.json` - 用户数据
  - `yelp_academic_dataset_business.json` - 商家数据

### 关键字段
```json
{
  "review_id": "评论ID",
  "user_id": "用户ID",
  "business_id": "商家ID",
  "stars": 5.0,
  "text": "评论文本",
  "date": "2023-01-01",
  "useful": 10,
  "funny": 2,
  "cool": 3
}
```

### 弱标签规则
基于以下6条规则构造（阈值≥4分标记为可疑）：
- 评分极端（1星或5星） +1分
- 评论长度异常 +1分
- 评论爆发性高 +2分
- 评分熵低 +2分
- 第一人称代词使用异常 +1分
- 情感极端 +1分

## 数据准备

### 下载数据集

1. **Amazon Reviews**
   - 下载地址：[Amazon Review Data](https://nijianmo.github.io/amazon/index.html)
   - 选择5-core版本
   - 解压到：`数据集/训练集/Amazon Review Data/`

2. **OpSpam**
   - 下载地址：[Op Spam AI](https://github.com/dbuscaldi/op_spam_AI)
   - 解压到：`数据集/训练集/op_spam_AI-main/`

3. **Yelp**
   - 下载地址：[Yelp Open Dataset](https://www.yelp.com/dataset)
   - 解压到：`数据集/测试集/Yelp-JSON/`

### 目录结构

```
数据集/
├── 训练集/
│   ├── Amazon Review Data/
│   │   ├── Electronics_5.json
│   │   ├── Books_5.json
│   │   └── ... (其他类别)
│   └── op_spam_AI-main/
│       ├── generated_GPT3_positive/
│       ├── generated_GPT3_negative/
│       ├── generated_LLama_positive/
│       └── generated_LLama_negative/
└── 测试集/
    └── Yelp-JSON/
        ├── yelp_academic_dataset_review.json
        ├── yelp_academic_dataset_user.json
        └── yelp_academic_dataset_business.json
```

## 数据统计

### 预期规模

| 数据集 | 记录数（完整） | 记录数（采样） | 文件大小 |
|--------|---------------|---------------|----------|
| Amazon | ~数百万 | 10,000 | ~数GB |
| OpSpam | 1,600 | 1,600 | ~2MB |
| Yelp | ~数百万 | 10,000 | ~数GB |

### 标准化后字段

所有数据集处理后统一为以下格式：

| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | str | 用户ID |
| item_id | str | 商品/商家ID |
| review_id | str | 评论ID（自动生成） |
| timestamp | datetime | 时间戳 |
| rating | float | 评分 |
| review_text | str | 评论文本 |
| platform | str | 平台标识 |
| weak_label | int | 弱标签（0=正常, 1=可疑） |
| label_source | str | 标签来源 |

## 注意事项

### 数据使用原则

1. **不混合数据集**：各数据集独立处理和分析
2. **弱标签非真实标签**：Amazon和Yelp的标签是启发式构造的
3. **OpSpam仅测试**：不参与任何参数学习或特征选择
4. **尊重数据许可**：仅用于学术研究

### 数据质量

- **缺失值**：部分字段可能为空（如timestamp、verified等）
- **编码问题**：统一使用UTF-8编码
- **数据清洗**：框架会自动处理异常值和缺失值

### 采样建议

首次运行建议启用采样模式：
```bash
python main.py --datasets all --sample
```

配置文件中可调整采样大小：
```yaml
sampling:
  enabled: true
  amazon_sample_size: 10000
  yelp_sample_size: 10000
```

## 数据引用

如果使用这些数据集发表论文，请引用原始数据源：

- **Amazon**: Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects.
- **Yelp**: Yelp Open Dataset (https://www.yelp.com/dataset)
- **OpSpam**: 根据您的具体数据源

