# Natural Language Processing with Disaster Tweets

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)

**一个清晰简洁的深度学习项目，用于灾难推文分类**

[竞赛链接](https://www.kaggle.com/competitions/nlp-getting-started) | [报告问题](https://github.com/RockDeng110/Natural-Language-Processing-with-Disaster-Tweets/issues)

</div>

---

## 📋 项目概述

这个项目解决Kaggle上的 **Natural Language Processing with Disaster Tweets** 竞赛。目标是预测一条推文是否关于真实灾难（1）或不是（0）。

### 🎯 项目目标

1. **学习重点**: 理解深度学习、RNN/LSTM架构和NLP工作流程
2. **应用批判性思维**: 在每个阶段做出有依据的决策
3. **取得好成绩**: 在时间和硬件限制下最大化F1分数

### 📊 数据集

- **训练集**: 7,613条推文（已标注）
- **测试集**: 3,263条推文（未标注）
- **特征**: `id`, `text`, `location`, `keyword`, `target`
- **评估指标**: F1 Score

---

## 🏗️ 项目结构（简化版）

```
Natural-Language-Processing-with-Disaster-Tweets/
│
├── data/
│   ├── raw/                    # 原始Kaggle数据
│   └── submissions/            # Kaggle提交文件
│
├── notebooks/
│   └── 00_complete_workflow_simple.ipynb    # 完整工作流程（从这里开始！）
│
├── requirements.txt            # Python依赖
├── .gitignore                 # Git忽略规则
└── README.md                  # 本文件
```

**设计理念**: 
- ✅ 简单直接 - 一个notebook包含完整流程
- ✅ 易于理解 - 所有代码在一处
- ✅ 快速开始 - 无需复杂设置
- ✅ 专注学习 - 将精力放在NLP和深度学习上

---

## 🚀 快速开始

### 1. **克隆仓库**

```bash
git clone https://github.com/RockDeng110/Natural-Language-Processing-with-Disaster-Tweets.git
cd Natural-Language-Processing-with-Disaster-Tweets
```

### 2. **安装依赖**

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖包
pip install -r requirements.txt

# 下载NLTK数据
python -c "import nltk; nltk.download('stopwords', quiet=True)"
```

### 3. **下载数据**

**方法1: 使用Kaggle API**
```bash
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip -d data/raw/
```

**方法2: 手动下载**
- 访问 [竞赛数据页面](https://www.kaggle.com/competitions/nlp-getting-started/data)
- 下载 `train.csv`, `test.csv`, `sample_submission.csv`
- 放到 `data/raw/` 目录

### 4. **运行Notebook**

```bash
jupyter notebook notebooks/00_complete_workflow_simple.ipynb
```

**就这么简单！** 从头到尾运行所有单元格，完成整个项目。

---

## 🧪 Notebook内容概览

`00_complete_workflow_simple.ipynb` 包含完整的端到端流程：

### 📑 1. 设置与配置
- 导入所有必要的库
- 配置超参数
- 检查GPU可用性

### 📊 2. 数据加载与EDA
- 加载训练和测试数据
- 分析目标分布（类别平衡）
- 检查文本长度统计
- 可视化数据特征

### 🔧 3. 文本预处理
- 清洗推文文本
- 处理URL、HTML标签
- 分词和序列化

### 📈 4. 基线模型
- TF-IDF向量化
- Logistic Regression训练
- 建立性能基准（F1 ≈ 0.75-0.78）

### 🧠 5. 深度学习模型
- 文本标记化和填充
- 双向LSTM架构
- 模型训练与验证
- 性能提升（F1 ≈ 0.78-0.82）

### 🎯 6. 结果与提交
- 模型对比分析
- 生成Kaggle提交文件
- 错误分析

### 📚 7. 学习总结
- 关键发现
- 批判性思考问题
- 后续改进方向

---

## 🧠 关键设计决策与理由

### 1. **文本预处理**

**决策**: 适度清洗（保留URL标记，保留标点符号）

**理由**:
- URL可能表示新闻来源（与灾难相关）
- 标点符号（!、?）可能表示紧急性
- 过度清洗会丢失有用信息

**替代方案**: 激进清洗（删除所有特殊字符）

**权衡**: 数据更干净但信息损失

---

### 2. **模型选择**

**决策**: 先尝试LSTM再考虑BERT

**理由**:
- 学习RNN基础（教育目标）
- 开发期间快速迭代
- 计算需求较低
- LSTM对短文本通常足够

**替代方案**: 直接跳到BERT

**权衡**: 更好的性能但学习机会减少

---

### 3. **超参数**

| 参数 | 值 | 理由 |
|------|-----|------|
| `max_sequence_length` | 128 | 覆盖95%+的推文，高效 |
| `embedding_dim` | 128 | 容量与速度的良好平衡 |
| `lstm_units` | 64 | 对小数据集足够 |
| `dropout_rate` | 0.5 | 防止过拟合 |
| `batch_size` | 32 | 适合GPU内存，训练稳定 |

---

### 4. **评估策略**

**决策**: F1分数作为主要指标，带阈值优化

**理由**:
- 竞赛指标是F1
- 比准确率更好地处理类别不平衡
- 阈值调整可以提升性能

**监控**: 同时跟踪精确率、召回率、AUC以获得更深入的见解

---

## 📈 预期结果

| 模型 | 验证F1 | 训练时间 | 参数量 |
|------|--------|---------|--------|
| TF-IDF + Logistic Regression | 0.75-0.78 | < 1分钟 | ~5K |
| 双向LSTM | 0.78-0.82 | ~10分钟 | ~200K |
| DistilBERT（可选） | 0.82-0.85 | ~30分钟 | ~66M |

*运行实验后填写实际结果*

---

## 💡 学到的经验

### 技术见解

1. **数据质量 > 模型复杂度**
   - 好的预处理比花哨的模型影响更大

2. **过拟合是真实存在的**
   - Dropout和早停是必不可少的
   - 验证集至关重要

3. **阈值调整很重要**
   - 默认的0.5很少是最优的
   - 简单的性能提升

### 流程学习

1. **从简单开始**
   - 基线快速建立基准
   - 有助于早期发现bug

2. **快速迭代**
   - 使用小数据子集进行快速实验
   - 只对最终模型进行完整训练

3. **跟踪所有内容**
   - Git用于代码版本控制
   - 实验日志记录结果
   - TensorBoard监控训练

---

## 🛠️ 开发流程

### 推荐的工作方式

1. **本地开发**（VS Code）
   - 编辑notebook
   - 进行版本控制（Git）
   - 代码审查和重构

2. **云端训练**（可选 - 如果需要GPU）
   - 使用Google Colab或Kaggle Notebooks
   - 利用免费GPU加速训练
   - 下载训练好的模型和结果

3. **提交结果**
   - 在Kaggle上生成submission.csv
   - 提交到竞赛平台

### Git使用建议

```powershell
# 初始化仓库
git init
git add .
git commit -m "初始项目设置"

# 开发过程中
git add <修改的文件>
git commit -m "描述性提交信息"
git push
```

---


## 🐛 调试技巧

### 常见问题与解决方案

**问题1**: 模型过拟合（训练损失 << 验证损失）
- ✅ 增加dropout率
- ✅ 添加更多正则化
- ✅ 减少模型容量
- ✅ 获取更多训练数据（数据增强）

**问题2**: 训练太慢
- ✅ 减少批量大小（如果内存不足）
- ✅ 使用GRU代替LSTM
- ✅ 减少序列长度
- ✅ 使用Colab/Kaggle GPU

**问题3**: 验证性能差
- ✅ 检查数据泄露（测试集在训练集中？）
- ✅ 验证预处理一致性
- ✅ 尝试不同架构
- ✅ 调整超参数

**问题4**: Notebook中的错误
- ✅ 使用`%debug`魔术命令进行事后调试
- ✅ 在函数中添加`print()`语句
- ✅ 在单独的单元格中测试小片段
- ✅ 重启内核并重新运行以清除状态

---

## 📚 资源与参考

### 有用的资料

- [Kaggle Tutorial](https://www.kaggle.com/philculliton/nlp-getting-started-tutorial) - 官方入门指南
- [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Chris Olah的博客
- [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/) - BERT可视化指南
- [TensorFlow Text Classification](https://www.tensorflow.org/tutorials/keras/text_classification) - 官方教程

### Papers

- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf) - 原始论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformers
- [BERT](https://arxiv.org/abs/1810.04805) - 预训练语言表示

### 数据科学工具

- [Kaggle Kernels](https://www.kaggle.com/code) - 浏览其他人的解决方案
- [Papers with Code](https://paperswithcode.com/task/text-classification) - 最新研究
- [Hugging Face](https://huggingface.co/models) - 预训练模型

---

## 🙏 致谢

- **Kaggle** - 举办竞赛
- **Figure-Eight** - 创建数据集
- **TensorFlow & scikit-learn** 团队 - 提供优秀工具
- **NLP社区** - 分享知识

---

## 📧 联系方式

项目链接: [https://github.com/YourUsername/Natural-Language-Processing-with-Disaster-Tweets](https://github.com/YourUsername/Natural-Language-Processing-with-Disaster-Tweets)

---

<div align="center">

Made with ❤️ 用于学习和成长

**如果这个项目对你有帮助，请给个Star ⭐**

</div>

```
