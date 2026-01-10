#!/bin/bash
# 依赖安装脚本 / Dependency Installation Script
# 用于安装 Marionette 项目及约束投影功能所需的所有依赖
# For installing all dependencies required by Marionette project and constraint projection feature

set -e  # 遇到错误立即退出 / Exit on error

echo "======================================"
echo "Marionette 依赖安装脚本"
echo "Marionette Dependency Installation"
echo "======================================"
echo ""

# 检测 Python 版本 / Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "检测到 Python 版本 / Detected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.9" && "$PYTHON_VERSION" != "3.8" ]]; then
    echo "警告: 推荐使用 Python 3.10 / Warning: Python 3.10 is recommended"
    echo "当前版本可能存在兼容性问题 / Current version may have compatibility issues"
fi

echo ""

# 提示用户选择安装方式 / Prompt user for installation method
echo "请选择安装方式 / Please choose installation method:"
echo "1) 完整安装 (推荐) - 使用 requirements1.txt / Full installation (Recommended) - Use requirements1.txt"
echo "2) 最小化安装 - 仅核心依赖 / Minimal installation - Core dependencies only"
echo "3) 仅安装缺失的依赖 / Install only missing dependencies"
echo ""
read -p "请输入选项 (1/2/3) / Enter option (1/2/3): " INSTALL_METHOD

# 检查 CUDA 是否可用 / Check if CUDA is available
echo ""
echo "检查 CUDA 可用性 / Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo "检测到 CUDA 版本 / Detected CUDA version: $CUDA_VERSION"
    USE_CUDA=true
else
    echo "未检测到 CUDA，将安装 CPU 版本 / CUDA not detected, will install CPU version"
    USE_CUDA=false
fi

echo ""

# 根据选择安装依赖 / Install dependencies based on choice
case $INSTALL_METHOD in
    1)
        echo "开始完整安装 / Starting full installation..."
        echo ""
        
        # 安装 PyTorch / Install PyTorch
        echo "步骤 1/2: 安装 PyTorch / Step 1/2: Installing PyTorch..."
        if [ "$USE_CUDA" = true ]; then
            pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
        else
            pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
        fi
        
        echo ""
        echo "步骤 2/2: 安装其他依赖 / Step 2/2: Installing other dependencies..."
        pip install -r requirements1.txt
        ;;
    2)
        echo "开始最小化安装 / Starting minimal installation..."
        echo ""
        
        # 仅安装核心依赖 / Install only core dependencies
        echo "安装核心依赖 / Installing core dependencies..."
        if [ "$USE_CUDA" = true ]; then
            pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
        else
            pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
        fi
        
        pip install numpy==1.26.4
        pip install einops==0.8.1
        pip install pytorch-lightning==1.9.5
        pip install hydra-core==1.3.2
        pip install omegaconf==2.3.0
        pip install pandas==2.2.2
        pip install tqdm==4.67.1
        pip install wandb==0.23.1
        ;;
    3)
        echo "检查并安装缺失的依赖 / Checking and installing missing dependencies..."
        echo ""
        
        # 检查每个核心依赖 / Check each core dependency
        MISSING_DEPS=()
        
        python3 -c "import torch" 2>/dev/null || MISSING_DEPS+=("torch")
        python3 -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")
        python3 -c "import einops" 2>/dev/null || MISSING_DEPS+=("einops")
        python3 -c "import pytorch_lightning" 2>/dev/null || MISSING_DEPS+=("pytorch_lightning")
        python3 -c "import hydra" 2>/dev/null || MISSING_DEPS+=("hydra-core")
        python3 -c "import omegaconf" 2>/dev/null || MISSING_DEPS+=("omegaconf")
        
        if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
            echo "所有核心依赖已安装 / All core dependencies are already installed"
        else
            echo "缺失的依赖 / Missing dependencies: ${MISSING_DEPS[*]}"
            echo "正在安装 / Installing..."
            
            for dep in "${MISSING_DEPS[@]}"; do
                case $dep in
                    "torch")
                        if [ "$USE_CUDA" = true ]; then
                            pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
                        else
                            pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
                        fi
                        ;;
                    "numpy")
                        pip install numpy==1.26.4
                        ;;
                    "einops")
                        pip install einops==0.8.1
                        ;;
                    "pytorch_lightning")
                        pip install pytorch-lightning==1.9.5
                        ;;
                    "hydra-core")
                        pip install hydra-core==1.3.2
                        ;;
                    "omegaconf")
                        pip install omegaconf==2.3.0
                        ;;
                esac
            done
        fi
        ;;
    *)
        echo "无效的选项 / Invalid option"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "验证安装 / Verifying Installation"
echo "======================================"
echo ""

# 验证核心依赖 / Verify core dependencies
echo "检查核心依赖 / Checking core dependencies..."
python3 << END
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError:
        print(f"✗ {display_name} - 缺失 / Missing")
        return False

success = True
success &= check_import('torch', 'PyTorch')
success &= check_import('numpy', 'NumPy')
success &= check_import('einops', 'einops')
success &= check_import('pytorch_lightning', 'PyTorch Lightning')
success &= check_import('hydra', 'Hydra')
success &= check_import('omegaconf', 'OmegaConf')

print("")
if success:
    print("所有核心依赖已成功安装! / All core dependencies installed successfully!")
    
    # 显示版本信息 / Show version info
    import torch
    print(f"\nPyTorch 版本 / version: {torch.__version__}")
    print(f"CUDA 可用 / available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本 / version: {torch.version.cuda}")
else:
    print("部分依赖缺失，请手动安装 / Some dependencies are missing, please install manually")
    sys.exit(1)
END

echo ""
echo "======================================"
echo "安装完成! / Installation Complete!"
echo "======================================"
echo ""
echo "下一步 / Next steps:"
echo "1. 运行训练 / Run training: python train.py"
echo "2. 运行采样 / Run sampling: sh sample_evaluation.sh <run_id>"
echo ""
echo "查看详细文档 / See detailed documentation:"
echo "- DEPENDENCIES.md - 依赖说明 / Dependency guide"
echo "- CONSTRAINT_PROJECTION_README.md - 约束投影使用 / Constraint projection usage"
echo "- PROJECTION_FREQUENCY_GUIDE.md - 投影频率指南 / Projection frequency guide"
echo ""
