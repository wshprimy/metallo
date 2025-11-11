# Metallo

## Overview

**Metallo** is a unified deep learning framework for **metal microstructure image analysis**. It provides standardized training, evaluation, and data preprocessing routines to accelerate research and deployment in computational materials science and optical imaging.

### Transformer库集成与数据流向分析

本仓库巧妙地集成了Hugging Face的`transformers`库，主要用于标准化模型训练流程和配置管理。以下是详细的数据和信息流向分析：

#### 1. 主要入口点 - train.py

[`train.py`](train.py) 是整个训练流程的核心入口点，它通过以下方式使用transformer库：

- **导入关键组件**：从`transformers`导入`Trainer`和`TrainingArguments`类
- **统一训练接口**：利用transformer的`Trainer`类提供标准化的训练、验证和测试流程
- **配置管理**：通过`TrainingArguments`管理所有训练超参数

#### 2. 数据流向架构

**数据输入层** → **数据处理层** → **模型编码层** → **融合预测层** → **输出层**

##### 2.1 数据输入与预处理
- **数据源**：[`MetalloDS`](metallo/data/unimetallo.py) 统一数据集类
  - 金属显微镜图像（PNG格式，300x300像素）
  - 光谱数据（每张图像对应24个光谱，每个光谱1600维）
  - DOS（态密度）标签值
- **数据分割策略**：基于slice_id的智能分割
  - slice_id="1"：95%训练集 + 5%验证集
  - slice_id="2"：100%测试集
- **图像变换**：标准化的torchvision变换管道
  ```python
  transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize(336),
      transforms.CenterCrop(300),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  ```

##### 2.2 模型架构与Transformer集成

**ToyNet模型** ([`metallo/models/toynet.py`](metallo/models/toynet.py))：
- 继承自`PreTrainedModel`，完全兼容transformer生态系统
- **图像编码器**：ResNet18/50骨干网络 → 特征维度压缩至hidden_dim
- **光谱编码器**：1D卷积网络处理24×1600光谱数据
- **融合策略**：简单拼接 + 多层感知机回归

**MetalloNet模型** ([`metallo/models/metallonet.py`](metallo/models/metallonet.py))：
- 高级版本，集成了BERT组件用于特征融合
- **空间注意力模块**：自适应权重分配机制
- **光谱编码器**：深度可分离卷积 + 空间重构（6×4网格）
- **Transformer融合**：`MetallographicDominantTransformerFusion`模块

##### 2.3 训练流程集成

```python
# 1. 配置初始化
training_args = TrainingArguments(**config.training.__dict__)

# 2. Trainer实例化
trainer = Trainer(
    model=model,                    # PreTrainedModel实例
    args=training_args,            # 训练配置
    train_dataset=datasets["train"], # 训练数据
    eval_dataset=datasets["eval"],   # 验证数据
    compute_metrics=st.compute_regression_metrics,  # 评估指标
    callbacks=callbacks,            # 自定义回调
)

# 3. 统一训练接口
trainer.train()                    # 开始训练
trainer.save_model()              # 模型保存
```

#### 3. 关键技术特性

##### 3.1 配置管理系统
- **PretrainedConfig继承**：`ToyNetConfig`和`MetalloNetConfig`继承自transformer的配置基类
- **参数序列化**：自动支持JSON/YAML配置文件的加载和保存
- **版本兼容性**：与Hugging Face模型库标准兼容

##### 3.2 模型保存与加载
```python
# 训练模式
model = MODEL_MAPPING[config.model.name]["model"](model_config)

# 测试模式 - 利用transformer的from_pretrained机制
model = MODEL_MAPPING[config.model.name]["model"](model_config).from_pretrained(
    config.mode.checkpoint_path, config=model_config
)
```

##### 3.3 回调系统集成
- **WandB日志记录**：[`metallo/callbacks/wandb_logger.py`](metallo/callbacks/wandb_logger.py)
- **训练损失监控**：[`metallo/callbacks/trainloss.py`](metallo/callbacks/trainloss.py)
- **检查点管理**：[`metallo/callbacks/checkpoint.py`](metallo/callbacks/checkpoint.py)

#### 4. 数据信息流总结

1. **输入阶段**：原始图像和光谱数据通过`MetalloDS`加载和预处理
2. **编码阶段**：双模态编码器分别提取图像和光谱特征
3. **融合阶段**：多模态特征通过注意力机制或简单拼接进行融合
4. **预测阶段**：回归头输出DOS预测值
5. **训练阶段**：transformer的`Trainer`管理整个训练循环
6. **评估阶段**：标准化的评估指标计算和结果输出

这种设计充分利用了transformer库的标准化训练流程，同时保持了对多模态金属材料数据的专业化处理能力，实现了科研代码的工程化和可复现性。

---

## Architecture

The project is structured into modular components under the `metallo` package:

```
metallo/
├── callbacks/        # Custom callbacks for monitoring, logging, and checkpoints
├── data/             # Data loading and preprocessing utilities
├── models/           # Model definitions (MetalloNet, ToyNet, etc.)
├── utils/            # Configuration, metrics, and helper functions
└── ...
```

### Core Modules

- [`metallo/models/metallonet.py`](metallo/models/metallonet.py): Defines **MetalloNet**, the main CNN model for microstructure recognition.
- [`metallo/models/toynet.py`](metallo/models/toynet.py): Includes **ToyNet**, a lightweight test model for quick experimentation.
- [`metallo/utils/config.py`](metallo/utils/config.py): Handles YAML-based config parsing and parameter management.
- [`metallo/data/unimetallo.py`](metallo/data/unimetallo.py): Defines the **UniMetallo dataset**, with automatic data normalization and augmentation.
- [`metallo/callbacks/wandb_logger.py`](metallo/callbacks/wandb_logger.py): Integrates **Weights & Biases (W&B)** logging for experiments.

---

## Installation & Development Setup（部署方式一）
两种部署方式选一个即可，直接部署或者使用docker部署。

### 1. Clone the Repository
从微信下载，可以省略这一步。

```bash
git clone https://github.com/your-org/metallo.git
cd metallo
```

### 2. Create a Python Environment
使用conda创建虚拟环境并管理，或者在之前的python环境基础上直接使用。

It’s recommended to use **Python 3.9+**.

```bash
conda create -n metallo python=3.9 -y
conda activate metallo
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Preprocessing

Prepare your raw dataset with:
```bash
python preprocess/preprocess.py --input <raw_data_dir> --output <processed_data_dir>
```

### 5. Start Training

You can start training by specifying one of the YAML config files (e.g., `metallonet_train.yaml`):

```bash
python train.py --config configs/metallonet_train.yaml
```

### 6. Run Evaluation

```bash
python train.py --config configs/toynet_eval.yaml
```

---

## Deployment (Docker-Based)

### Build Docker Image

```bash
bash scripts/autobuild.sh
```

or manually:

```bash
docker build -t metallo:latest .
```

### Start a Docker Container

```bash
bash scripts/start_docker.sh
```

This launches a container with GPU support (if available) and mounts the local directory for immediate code access.

---

## Detailed Deployment Guide (for Optical Engineering Students)

This section provides a step-by-step process assuming **no deep learning background** is required.

### Step 1. Install Docker
无论是Windows, macOS, 还是 Linux，都建议在docker官网安装。
- **Windows/macOS:** Download from [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux (Ubuntu):**
  ```bash
  sudo apt update
  sudo apt install docker.io
  sudo systemctl enable docker
  sudo systemctl start docker
  sudo usermod -aG docker $USER
  ```

Then log out and log back in to activate permissions.

### Step 2. Verify Installation

```bash
docker --version
```

### Step 3. Build the Project Image

Run the following from your project root:

```bash
bash scripts/autobuild.sh
```

This step compiles a container with all Python and CUDA dependencies set up automatically.

### Step 4. Run the Container

```bash
bash scripts/start_docker.sh
```

Inside the container, you’ll see a prompt such as:
```
root@metallo-container:/workspace/metallo#
```

### 4.9 Run Preprocessing
记得先预处理数据集

Prepare your raw dataset with:
```bash
python preprocess/preprocess.py --input <raw_data_dir> --output <processed_data_dir>
```

### Step 5. Train a Model Inside Docker
更建议使用scripts下的脚本，例如train.sh等...

Run:
```bash
python train.py --config configs/metallonet_train.yaml
```

You can monitor progress in W&B if you have an account configured.

### Step 6. Evaluate a Model

```bash
python train.py --config configs/toynet_eval.yaml
```

### Step 7. Export or Visualize Results

All training outputs (e.g., checkpoints, logs) will appear under:
```
outputs/checkpoints/
outputs/logs/
```

You can visualize these with:
```bash
bash scripts/eval.sh
```

---

## License
This project is distributed under the **MIT License**. See [LICENSE](LICENSE) for more details.
