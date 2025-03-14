
# DTIBFAI-Drug-target-Interaction-Prediction-Based-on-BERT-and-Feature-Augment-of-Informer

## 关于我们 About US
药物-靶标相互作用（DTIs）对药物发现至关重要，为新疗法提供了见解。语言模型的发展为DTI预测提供了有力支持。然而，基于transformer的自注意力机制往往无法捕获细粒度的药物靶标相互作用，单模态特征表示限制了完全表征DTIs的能力。针对上述问题，提出了一种基于BERT和inforformer模型的预测方法DTIBFAI。DTIBFAI使用BioBERT和ChemBERTa对药物和蛋白质序列进行预处理，同时结合分子指纹和双肽组成特征以增强特征的丰富性和代表性。此外，该研究集成了一种改进的告密器用于DTI预测问题。改进后的告密者增强了特征嵌入，有效地捕获了序列数据内部复杂的交互模式。将DTIBFAI模型与几种最新的药物-靶标作用关系预测方法进行比较，评估其性能。实验结果表明，DTIBFAI在评测数据集上的AUROC和AUPRC分数分别达到0.9661和0.9673，明显优于上述方法。案例研究揭示了该模型识别新型DTIs的能力，包括以前未记录的相互作用，并通过生物学合理性验证。这些结果证明了DTIBFAI在推进DTI预测及其在药物发现中的应用方面的潜力。

Drug-target interactions (DTIs) are critical for drug discovery, providing insights into novel therapies. The development of language models has provided strong support for DTI prediction. However, Transformer-based self attention mechanisms often fail to capture fine grained drug target interactions, and single-mode feature representations limit the ability to fully characterize DTIs. To address these issues, this study proposes a novel prediction method DTIBFAI based on BERT and Informer model. DTIBFAI preprocesses drug and protein sequences using BioBERT and ChemBERTa, while incorporating molecular fingerprints and dipeptide composition features to augment the richness and representativeness of the features. Additionally, this study integrates a modified Informer for the DTI prediction problem. The modified Informer augments feature embeddings and effectively captures the complex interaction patterns within sequence data. The performance of the DTIBFAI model was evaluated by comparing it with several state-of-the-art methods for drug-target interaction prediction. Experimental results demonstrated that DTIBFAI significantly outperformed these methods on the evaluated datasets, achieving AUROC and AUPRC scores of 0.9661 and 0.9673, respectively. Case studies reveal the model's ability to identify novel DTIs, including previously unrecorded interactions, validated for their biological plausibility. These results demonstrate the potential of DTIBFAI in advancing DTI prediction and its application in drug discovery.

## ⚠️⚠️ ​重要提示 Important Notice ⚠️⚠️ 

**这个项目需要存储库中没有包含的其他文件 
This project requires additional files that are not included in the repository.**  
**请在项目运行前检查项目目录是否完整 
Please check that the project directory is complete before running the project**

在运行模型前请先下载下列文件：
Please download the following resources before proceeding:  
- 预训练模型    Pre-trained models    (抱脸 huggingface): [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) | [ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)
- 数据集压缩包    Dataset files    (百度网盘 Baidu Webdisk): [Full Dataset Package](https://pan.baidu.com/s/1TNyLJSUbYj0lGsQfLDdjTA?pwd=4jxz)
- 
- (我们也提供基于百度网盘的预训练模型下载，您可以通过以下链接下载)
- 百度网盘存储的预训练模型 Pre-trained models Download Based on Baidu WebDisk(百度网盘 Baidu Webdisk): [Pre-trained models](https://pan.baidu.com/s/1PfUJuOnSrMe0mjqcrjqsag?pwd=4kex)


## 🚀 功能特性 Functional features
- **双模态编码**：结合BioBERT蛋白质编码与ChEMBERT药物分子编码
- **特征融合**：整合ECFP分子指纹与二肽组成特征
- **高效训练**：引入ProbAttention降低模型内存和时间开销
- **评估指标**：AUC-ROC、AUPR、F1-Score等综合评估体系


## 🗂 项目结构 Project Structure（请在项目运行前检查项目结构是否完整、文件夹命名是否正确）
```bash
.
├── biobert-v1.2/            # BioBERT预训练模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── chamberts/               # ChEMBERT预训练模型
│   ├── config.json
│   ├── ...                  # （ChemBERTa预训练模型文件）
│   └── pytorch_model.bin
├── dataset/                 # 数据存储
│   └── sBioSNAP/
│      ├── test.csv          # 测试集
│      ├── val.csv           # 验证集
│      ├── train.csv         # 训练集
│      └── all.csv           # 总数据集
├── models.py                # 模型定义
├── config.py                # 超参数配置
├── stream.py                # 数据预处理
├── attn.py                  # probattention注意力模块
├── decoder.py               # 解码器
├── encoder.py               # 编码器
├── embed.py                 # 嵌入器
├── setup.py                 # 项目描述
├── train.py                 # 主训练脚本
└── output.txt               # 模型在运行后生成的训练日志 （⚠️可空缺 ⚠️Available for vacancy）
```


## 📦 环境依赖 Environment dependencies
- python=3.12.9
- cuda-version=12.8=3
- cudnn=9.1.1.17=cuda12_1
- pytorch=2.5.1=py3.12_cuda12.1_cudnn9.1.0_0
- transformers=4.45.2
- scikit-learn=1.6.1
- pandas=2.2.3
- numpy=1.26.4
- numpy-base=1.26.4

安装依赖：
```bash
pip install numpy==1.26.4
```


## 📂 数据准备 Data preparation
1. 下载预训练模型（huggingface）：
   - [BioBERT-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/main) 放置于 `biobert-v1.2/`
   - [ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) 放置于 `chamberts/`

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


## 🏃 训练流程 Training flow
```bash
python train.py
```

关键参数 Key parameters：
- `--batch-size`: 批次大小 (默认：16)
- `--epochs`: 训练轮次 (默认：50)
- `--lr`: 初始学习率 (默认：1e-5)


## 📜 许可证
本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE)


## 🙏 致谢
- BioBERT: [DMIS-Lab](https://dmis.korea.ac.kr/)
- ChemBERTa: [DeepChem](https://deepchem.io/)

