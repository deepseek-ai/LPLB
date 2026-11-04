import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA GPU')
def test_jit_uses_cuda_home_including_cccl(tmp_path: Path) -> None:
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if not cuda_home:
        nvcc = shutil.which('nvcc')
        cuda_home = str(Path(nvcc).resolve().parent.parent) if nvcc else '/usr/local/cuda'
    if not (Path(cuda_home) / 'include').is_dir():
        pytest.skip('CUDA toolkit include directory is unavailable')

    # A different spelling of the toolkit path makes the selected include
    # directories observable in the JIT cache metadata on standard installs too.
    toolkit = tmp_path / 'toolkit'
    toolkit.symlink_to(Path(cuda_home).resolve(), target_is_directory=True)
    cache = tmp_path / 'cache'
    env = os.environ.copy()
    env.update(CUDA_HOME=str(toolkit), LPLB_CACHE_PATH=str(cache))
    program = (
        'from lplb import Planner\n'
        'from tests.utils import CUBE_8P2E\n'
        'Planner(CUBE_8P2E, 272, 256, ep_size=8)\n'
    )
    subprocess.run([sys.executable, '-c', program], env=env, check=True, timeout=300)  # noqa: S603

    option_files = list((cache / 'cache').glob('*/options'))
    assert len(option_files) == 1
    options = option_files[0].read_text().splitlines()
    assert f'-I{toolkit}/include' in options
    assert f'-I{toolkit}/include/cccl' in options
