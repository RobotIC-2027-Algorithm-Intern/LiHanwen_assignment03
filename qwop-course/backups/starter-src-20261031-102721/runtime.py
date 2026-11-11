"""本教程使用 PyTorch；让 TensorBoard 使用自带兼容层，避免导入本机 TensorFlow。"""
import sys
import types
from pathlib import Path

# Optional teacher-local CPU build: avoid loading CUDA libraries on a machine
# with limited memory. Ubuntu setup already installs CPU PyTorch in its venv.
cpu_build = Path(__file__).resolve().parent / 'reference/torch-cpu'
if (cpu_build / 'READY').exists():
    sys.path.insert(0, str(cpu_build))

# TensorBoard compat 官方加载逻辑会用 notf 模块作为无需 TensorFlow 的标记。
sys.modules.setdefault('tensorboard.compat.notf', types.ModuleType('tensorboard.compat.notf'))
