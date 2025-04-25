import math

import numpy as np

from camvidlog2.ai import get_string_embedding


def test_get_test_embeddings():
    embedding = get_string_embedding("hello world")

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 512
    assert embedding.dtype == np.float32
    assert math.isclose(embedding.sum(), 1.0)
