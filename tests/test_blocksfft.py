import numpy as np
from modalpy.core.base import blocksfft


def test_blocksfft_constant_signal():
    q = np.ones((4, 2))
    result = blocksfft(q, nfft=4, nblocks=1, novlap=0, n_threads=2)
    assert result.shape == (4 // 2 + 1, 2, 1)
    assert np.allclose(result, 0)

