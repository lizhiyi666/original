# 依赖安装说明 / Dependency Installation Guide

## 问题说明 / Problem Description

在运行新增的约束投影功能时，可能会遇到 "No module named 'xxx'" 的错误。本文档说明了所需的依赖。

When running the new constraint projection feature, you may encounter "No module named 'xxx'" errors. This document explains the required dependencies.

## 核心依赖 / Core Dependencies

### 本实现新增的导入 / New Imports Added by This Implementation

我们的实现（`constraint_projection.py` 和修改的 `diffusion_transformer.py`）**没有引入任何新的外部依赖**。所有使用的库都是原项目已有的：

Our implementation (`constraint_projection.py` and modified `diffusion_transformer.py`) **does not introduce any new external dependencies**. All libraries used are already present in the original project:

1. **torch** - PyTorch 核心库 / PyTorch core library
2. **torch.nn.functional** - PyTorch 神经网络功能模块 / PyTorch neural network functional module
3. **numpy** - 数值计算库 / Numerical computing library
4. **einops** - 张量操作库 / Tensor operation library

这些都已包含在原始项目的 `requirements.txt` 和 `requirements1.txt` 中。

These are all already included in the original project's `requirements.txt` and `requirements1.txt`.

## 完整依赖列表 / Complete Dependency List

### 方法 1: 使用 requirements1.txt (推荐) / Method 1: Use requirements1.txt (Recommended)

项目包含一个完整的依赖列表文件 `requirements1.txt`，包含所有必需的包及其版本：

The project includes a complete dependency list file `requirements1.txt` with all required packages and their versions:

```bash
pip install -r requirements1.txt
```

**关键依赖 / Key Dependencies:**
- `torch==2.0.0` - PyTorch 深度学习框架
- `numpy==1.26.4` - 数值计算
- `pytorch-lightning==1.9.5` - PyTorch Lightning 训练框架
- `hydra-core==1.3.2` - 配置管理
- `wandb==0.23.1` - 实验跟踪
- `einops==0.8.1` - 张量操作
- `recbole==1.2.1` - 推荐系统评估
- `tqdm==4.67.1` - 进度条
- 其他 CUDA 相关库 / Other CUDA-related libraries

### 方法 2: 使用 requirements.txt (最小化) / Method 2: Use requirements.txt (Minimal)

如果只想安装核心依赖，可以使用简化的 `requirements.txt`：

If you only want to install core dependencies, use the simplified `requirements.txt`:

```bash
pip install -r requirements.txt
```

然后补充安装 PyTorch：
Then supplement with PyTorch installation:

```bash
pip install torch==2.0.0
```

**注意 / Note:** `requirements.txt` 中某些包未指定版本，可能导致版本冲突。

Some packages in `requirements.txt` don't specify versions, which may cause version conflicts.

## 分步安装指南 / Step-by-Step Installation Guide

### 步骤 1: 创建虚拟环境 (推荐) / Step 1: Create Virtual Environment (Recommended)

```bash
# 使用 conda
conda create -n marionette python=3.10
conda activate marionette

# 或使用 venv
python3.10 -m venv marionette_env
source marionette_env/bin/activate  # Linux/Mac
# 或 marionette_env\Scripts\activate  # Windows
```

### 步骤 2: 安装 CUDA 工具包 (如需 GPU 支持) / Step 2: Install CUDA Toolkit (for GPU Support)

确保系统已安装 CUDA 11.7：
Ensure CUDA 11.7 is installed on your system:

```bash
# 检查 CUDA 版本 / Check CUDA version
nvcc --version
```

如果未安装，请访问 NVIDIA 官网下载 CUDA 11.7。
If not installed, download CUDA 11.7 from NVIDIA's official website.

### 步骤 3: 安装 PyTorch / Step 3: Install PyTorch

```bash
# CUDA 11.7 版本 / For CUDA 11.7
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# 或 CPU 版本 / Or CPU version
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

### 步骤 4: 安装其他依赖 / Step 4: Install Other Dependencies

```bash
# 使用完整的依赖列表 / Use complete dependency list
pip install -r requirements1.txt

# 如果遇到某些包安装失败，可以跳过它们 / If some packages fail, you can skip them
pip install -r requirements1.txt --no-deps
# 然后手动安装关键依赖 / Then manually install key dependencies
```

### 步骤 5: 验证安装 / Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pytorch_lightning; print('PyTorch Lightning OK')"
python -c "import einops; print('einops OK')"
python -c "import numpy; print('numpy OK')"
```

## 常见问题及解决方案 / Common Issues and Solutions

### 问题 1: No module named 'torch'

**解决方案 / Solution:**
```bash
pip install torch==2.0.0
```

### 问题 2: No module named 'einops'

**解决方案 / Solution:**
```bash
pip install einops==0.8.1
```

### 问题 3: No module named 'pytorch_lightning'

**解决方案 / Solution:**
```bash
pip install pytorch-lightning==1.9.5
```

### 问题 4: No module named 'hydra'

**解决方案 / Solution:**
```bash
pip install hydra-core==1.3.2
```

### 问题 5: No module named 'recbole'

**解决方案 / Solution:**
```bash
pip install recbole==1.2.1
```

### 问题 6: CUDA 版本不匹配

如果遇到 CUDA 相关错误：

**解决方案 / Solution:**
```bash
# 检查系统 CUDA 版本 / Check system CUDA version
nvcc --version

# 根据 CUDA 版本安装对应的 PyTorch / Install PyTorch based on CUDA version
# CUDA 11.7
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

### 问题 7: RecBole LightGCN 依赖问题

根据 README.md 的说明，RecBole 中的 LightGCN 有已废弃的依赖：

**解决方案 / Solution:**
参考：https://github.com/RUCAIBox/RecBole/issues/2090

或选择其他下游任务代替 LightGCN。

## 最小化依赖 (仅运行约束投影) / Minimal Dependencies (Only for Constraint Projection)

如果你只想运行约束投影功能（不包括完整的训练和评估），可以只安装以下依赖：

If you only want to run the constraint projection feature (without full training and evaluation), you can install only these dependencies:

```bash
pip install torch==2.0.0
pip install numpy==1.26.4
pip install einops==0.8.1
pip install pytorch-lightning==1.9.5
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0
```

## 依赖版本兼容性 / Dependency Version Compatibility

**Python 版本 / Python Version:**
- 推荐 / Recommended: Python 3.10
- 支持 / Supported: Python 3.8 - 3.11

**PyTorch 版本 / PyTorch Version:**
- 项目使用 / Project uses: PyTorch 2.0.0
- CUDA: 11.7

**注意事项 / Notes:**
1. PyTorch 2.0.0 需要 CUDA 11.7 或更高版本（如使用 GPU）
2. 某些依赖可能与 Python 3.12 不兼容
3. Windows 用户可能需要安装 Microsoft Visual C++ 14.0 或更高版本

## 快速启动脚本 / Quick Start Script

创建一个 `setup.sh` 文件：
Create a `setup.sh` file:

```bash
#!/bin/bash

# 创建虚拟环境 / Create virtual environment
conda create -n marionette python=3.10 -y
conda activate marionette

# 安装 PyTorch (CUDA 11.7) / Install PyTorch (CUDA 11.7)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# 安装其他依赖 / Install other dependencies
pip install -r requirements1.txt

# 验证安装 / Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import pytorch_lightning; import einops; import numpy; print('All core dependencies installed successfully!')"

echo "Setup complete! Activate environment with: conda activate marionette"
```

运行脚本 / Run script:
```bash
chmod +x setup.sh
./setup.sh
```

## 总结 / Summary

**本实现没有引入新的依赖**，所有使用的库都在原项目的依赖列表中。如果遇到模块缺失错误：

**This implementation does not introduce new dependencies**. All libraries used are in the original project's dependency list. If you encounter module missing errors:

1. **首选方案 / Preferred**: 使用 `pip install -r requirements1.txt` 安装完整依赖
2. **最小化方案 / Minimal**: 安装 torch, numpy, einops, pytorch-lightning, hydra-core
3. **逐个安装 / Individual**: 根据错误提示逐个安装缺失的包

如有问题，请检查：
- Python 版本是否为 3.10
- CUDA 版本是否为 11.7（如使用 GPU）
- 是否在虚拟环境中安装

If issues persist, check:
- Python version is 3.10
- CUDA version is 11.7 (if using GPU)
- Installation is in a virtual environment
