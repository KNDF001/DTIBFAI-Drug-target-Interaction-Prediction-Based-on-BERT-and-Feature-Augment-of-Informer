```markdown
# Drug-Target Interaction Prediction

基于深度学习的药物-靶标相互作用预测模型，整合蛋白质序列特征与药物分子表示

## 🚀 功能特性
- **双模态编码**：结合BioBERT蛋白质编码与ChEMBERT药物分子编码
- **特征融合**：整合ECFP分子指纹与二肽组成特征
- **高效训练**：支持GPU加速与预训练模型冻结
- **评估指标**：AUC-ROC、AUPR、F1-Score等综合评估体系

## 📦 环境依赖
- Python 3.8+
- PyTorch 1.12+
- Transformers 4.28+
- scikit-learn 1.2+
- pandas 1.5+
- numpy 1.23+

安装依赖：
```bash
pip install torch transformers scikit-learn pandas numpy
```

## 📂 数据准备
1. 下载预训练模型：
   - [BioBERT-v1.2](https://huggingface.co/dmis-lab/biobert-v1.2) 放置于 `biobert-v1.2/`
   - [ChEMBERT](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) 放置于 `chamberts/`

2. 数据集结构：
```bash
dataset/
└── sBioSNAP/
    ├── all.csv      # 完整数据集
    ├── train.csv    # 训练集
    ├── val.csv      # 验证集
    └── test.csv     # 测试集
```

数据集应包含以下列：
- `SMILES`: 药物分子SMILES表示
- `Target Sequence`: 蛋白质氨基酸序列
- `ECFP`: 扩展连通性指纹（1024维）
- `Label`: 相互作用标签 (0/1)

## 🧠 模型架构
```python
class BIN_Interaction_Flat(nn.Module):
    def __init__(self, **config):
        super().__init__()
        # 蛋白质编码器
        self.protein_encoder = nn.Linear(768, 256)
        # 药物编码器  
        self.drug_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        # 交互预测层
        self.predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

## 🏃 训练流程
```bash
python train.py \
  --batch-size 32 \
  --epochs 50 \
  --lr 1e-5 \
  --workers 4
```

关键参数：
- `--batch-size`: 批次大小 (默认：16)
- `--epochs`: 训练轮次 (默认：50)
- `--lr`: 初始学习率 (默认：1e-5)
- `--workers`: 数据加载线程数


## 🗂 项目结构
```bash
.
├── biobert-v1.2/            # BioBERT预训练模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── chamberts/               # ChEMBERT预训练模型
│   ├── config.json
│   └── pytorch_model.bin
├── models/                  # 模型定义
│   └── BIN_Interaction_Flat.py
├── dataset/                 # 数据存储
│   └── sBioSNAP/
├── config.py                # 超参数配置
├── stream.py                # 数据预处理
├── train.py                 # 主训练脚本
└── output.txt               # 训练日志
```

## 📜 许可证
本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE)

## 🙏 致谢
- BioBERT: [DMIS-Lab](https://github.com/dmis-lab/biobert)
- ChEMBERT: [DeepChem](https://github.com/deepchem/deepchem)
```
