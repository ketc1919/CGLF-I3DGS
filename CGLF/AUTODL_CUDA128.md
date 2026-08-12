# AutoDL：CUDA 12.8 环境配置

当服务器只提供 CUDA 12.8 编译器时，不要使用原始 `environment.yml` 中的 PyTorch 1.12.1 / CUDA 11.6 组合。CUDA 扩展的编译器主版本必须与 PyTorch CUDA 主版本一致。

下面的环境与本仓库 I3DGS 使用的计算平台保持一致，但安装在独立 Conda 环境中：

```bash
mkdir -p /root/autodl-tmp/conda-envs
mkdir -p /root/autodl-tmp/conda-pkgs
mkdir -p /root/autodl-tmp/pip-cache
mkdir -p /root/autodl-tmp/build-tmp
mkdir -p /root/autodl-tmp/torch-cache

export CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export TMPDIR=/root/autodl-tmp/build-tmp
export TORCH_HOME=/root/autodl-tmp/torch-cache
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.6

conda create --prefix /root/autodl-tmp/conda-envs/cglf-cu128 python=3.12 pip -y
conda activate /root/autodl-tmp/conda-envs/cglf-cu128

python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-cu128.txt

python -m pip install --no-index torch-scatter \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

python -m pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install -v --no-build-isolation ./submodules/simple-knn
```

验证：

```bash
python - <<'PY'
import torch
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizer

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("CGLF CUDA extensions: OK")
PY
```

如果扩展安装失败，请保留完整日志，并优先提供从 `error:` 开始到日志结尾的内容。
