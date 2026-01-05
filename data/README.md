# 数据目录

本目录用于存放原始数据和处理后的数据。

## 目录结构

```
data/
├── raw/              # 原始数据（请将数据集放在这里）
│   ├── amazon/
│   ├── opspam/
│   └── yelp/
│
└── processed/        # 处理后的数据（由Pipeline自动生成）
    ├── amazon/
    ├── opspam/
    └── yelp/
```

## 数据集获取

请参阅 [DATASET.md](../DATASET.md) 了解如何获取和准备数据集。

## 注意事项

- 原始数据文件较大，已添加到 `.gitignore`，不会上传到Git仓库
- 处理后的数据也不会上传，需要本地运行Pipeline生成

