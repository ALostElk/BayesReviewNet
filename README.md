# BayesReviewNet

**基于贝叶斯网络的虚假评论识别框架**

本项目不使用深度学习模型，而是采用概率图模型（Probabilistic Graphical Model）方法，通过贝叶斯网络对评论数据进行建模与推断。

---

## 项目特点

- **方法论导向**: 清晰的贝叶斯网络建模流程，而非黑盒分类器
- **可解释性**: 所有特征都是可解释的统计量，DAG结构反映因果假设
- **模块化设计**: 分层架构，职责明确，便于复现和扩展
- **弱监督学习**: 使用启发式规则和平台信号，不依赖大规模标注数据

---

## 代码结构

本项目采用**分层模块化**设计，每一层对应贝叶斯网络建模的不同阶段：

```
BayesReviewNet/
├── src/
│   ├── preprocessing/           # 【第1层】数据预处理
│   │   ├── amazon.py            #   - Amazon数据加载与标准化
│   │   ├── yelp.py              #   - Yelp数据加载与标准化
│   │   ├── opspam.py            #   - OpSpam数据加载（仅测试）
│   │   └── common.py            #   - 通用清洗函数
│   │
│   ├── features/                # 【第2层】特征工程
│   │   ├── text_features.py     #   - 文本统计特征（非embedding）
│   │   ├── behavior_features.py #   - 用户行为特征
│   │   └── discretize.py        #   - 连续变量离散化
│   │
│   ├── bayes/                   # 【第3层】贝叶斯网络核心
│   │   ├── variables.py         #   - 随机变量定义与取值空间
│   │   ├── structure.py         #   - DAG结构定义（因果关系）
│   │   ├── cpds.py              #   - 条件概率表学习
│   │   └── inference.py         #   - 概率推断（非分类）
│   │
│   ├── evaluation/              # 【第4层】评估
│   │   ├── metrics.py           #   - Precision/Recall/F1/ROC-AUC
│   │   └── test_sets.py         #   - OpSpam/YelpChi测试集处理
│   │
│   ├── utils/                   # 工具模块
│   │   ├── io.py
│   │   ├── logging.py
│   │   └── config.py
│   │
│   └── main.py                  # Pipeline调度器（唯一入口）
│
├── configs/
│   └── default.yaml             # 配置文件
│
├── tests/                       # 单元测试
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_discretization.py
│
├── data/                        # 数据目录（.gitignore）
│   ├── raw/                     #   - 原始数据
│   └── processed/               #   - 处理后数据
│
├── README.md
└── requirements.txt
```

---

## 方法流程

### 第1层: 数据预处理 (Preprocessing)

**职责**: 仅负责数据加载与字段标准化，不涉及特征计算

- 统一字段映射: `reviewerID` → `user_id`, `asin` → `item_id`
- 时间戳标准化
- 文本清洗（移除控制字符、多余空格）
- 输出: 标准化的 DataFrame

**原则**:
- ❌ 不计算复杂特征
- ❌ 不进行标签推断
- ✅ 只做格式统一

### 第2层: 特征工程 (Feature Engineering)

**职责**: 产生可解释的统计特征，并离散化为有限状态

**文本特征** (`text_features.py`):
- 评论长度、词数、句子数
- 情感分数（TextBlob）
- 主观性分数
- 感叹号密度、第一人称代词比例

**行为特征** (`behavior_features.py`):
- 用户评论数、平均评分、评分标准差
- 评分偏差（相对全局均值）
- 评分熵（多样性）
- 评论爆发性（时间间隔）

**离散化** (`discretize.py`):
- 将连续特征映射为离散状态（如 LOW/MEDIUM/HIGH）
- 适用于贝叶斯网络的CPD学习

**原则**:
- ❌ 禁止深度学习、预训练模型、embedding
- ✅ 只使用统计方法
- ✅ 所有特征必须可解释

### 第3层: 贝叶斯网络 (Bayesian Network)

**职责**: 贝叶斯网络建模的核心模块

**变量定义** (`variables.py`):
- 定义所有随机变量及其取值空间
- 示例: `REVIEW_LENGTH = BayesianVariable(name='review_length_discrete', states=['SHORT', 'NORMAL', 'LONG'])`

**结构定义** (`structure.py`):
- 定义DAG（有向无环图）
- 明确节点间的因果关系
- 示例: `user_review_count → weak_label`

**CPD学习** (`cpds.py`):
- 从数据中学习条件概率表 P(X | Parents(X))
- 使用Laplace平滑防止零概率

**推断** (`inference.py`):
- 计算后验概率 P(weak_label | evidence)
- 支持Markov Blanket解释
- ❌ 不做分类决策，只输出概率

**原则**:
- ✅ 明确区分随机变量、DAG、CPD、推断
- ✅ 支持结构学习和因果推断
- ✅ 便于后续Markov Blanket分析

### 第4层: 评估 (Evaluation)

**职责**: 事后验证，不参与训练

**指标** (`metrics.py`):
- Precision, Recall, F1-Score
- ROC-AUC
- 混淆矩阵

**测试集** (`test_sets.py`):
- OpSpam: 具有ground truth标签（LLM生成的虚假评论）
- YelpChi: 使用平台过滤标签（noisy weak labels）

**原则**:
- ❌ 测试集绝不参与训练
- ✅ 明确区分推断概率和分类阈值

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行Pipeline

```bash
# 处理所有数据集（默认结构）
python src/main.py

# 处理特定数据集
python src/main.py --datasets amazon opspam

# 使用朴素贝叶斯结构
python src/main.py --structure naive

# 指定配置文件
python src/main.py --config configs/default.yaml
```

### 3. 运行测试

```bash
# 运行所有测试
python -m unittest discover tests

# 运行特定测试
python -m unittest tests.test_preprocessing
```

---

## 数据集

本项目使用三个数据集:

1. **Amazon Reviews (5-core)**: 训练集，使用启发式规则构造弱标签
2. **OpSpam (AMT)**: 测试集，具有ground truth标签
3. **Yelp Open Dataset**: 测试集，使用平台过滤标签

详细说明请参阅 [DATASET.md](DATASET.md)

---

## 配置文件

配置文件位于 `configs/default.yaml`，包含:

- 数据路径
- 特征工程参数
- 离散化规则（bins & labels）
- 采样配置

示例:

```yaml
discretization:
  review_length:
    bins: [0, 50, 200, 10000]
    labels: ["SHORT", "NORMAL", "LONG"]
  
  sentiment_score:
    bins: [-1.0, -0.3, 0.3, 1.0]
    labels: ["NEGATIVE", "NEUTRAL", "POSITIVE"]
```

---

## 输出文件

Pipeline运行后会生成以下文件:

```
processed_data/           # 标准化数据
discretized/              # 最终数据（包含离散化特征和后验概率）
metadata/                 # 元数据（特征描述、离散化规则、评估结果）
logs/                     # 日志文件
```

---

## 研究价值

本项目的结构设计直接对应论文的**方法章节**:

1. **数据预处理**: 对应论文的"数据描述"
2. **特征工程**: 对应"特征设计"章节
3. **贝叶斯网络**: 对应"模型构建"章节
   - DAG结构 → 因果假设
   - CPD学习 → 参数估计
   - 推断 → 后验概率计算
4. **评估**: 对应"实验结果"章节

审稿人可以清晰地看到:
- ✅ 这是一个贝叶斯网络项目，而非深度学习项目
- ✅ 每一步都有明确的概率论解释
- ✅ 支持Markov Blanket、d-separation等图模型分析

---

## 依赖库

- `pandas`, `numpy`: 数据处理
- `scipy`: 熵计算
- `textblob`: 情感分析
- `networkx`: 图结构处理
- `scikit-learn`: 评估指标
- `pyyaml`: 配置文件
- `fastparquet`: Parquet文件支持

**注意**: 不依赖 `torch`, `tensorflow`, `transformers` 等深度学习库

---

## 常见问题

**Q: 为什么不使用深度学习?**

A: 本项目专注于概率图模型方法论，强调可解释性和因果推断。深度学习是黑盒模型，不符合研究目标。

**Q: 弱标签如何构造?**

A: 使用启发式规则（如"未验证购买 + 评分极端 + 评论爆发性高"）或平台过滤标签，不依赖大规模人工标注。

**Q: 如何扩展到其他数据集?**

A: 在 `src/preprocessing/` 下创建新的预处理器，继承通用接口即可。

---

## 许可证

本项目仅用于学术研究，请遵守数据集的原始许可证。

---

## 引用

如果本项目对您的研究有帮助，请引用相关数据集:

- Amazon Reviews: [引用信息]
- OpSpam: [引用信息]
- Yelp Open Dataset: [引用信息]
