"""Use the established sampler with the FP32/TF32 policy verified by perfcal-v1."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    runpy.run_module('sample', run_name='__main__')
