from __future__ import division, print_function
import numpy as np
from torch_submod.blocks import blockwise_means, blocks_2d


def test_2d_blocks():
    matrix = np.asarray([
        [-1, -3, 4, 5, 1, 1, 1],
        [-1, -1, 4, 4, 4, 1, 1]], dtype=np.float32)
    blocks = np.asarray([
        [0, 1, 2, 3, 4, 4, 4],
        [0, 0, 2, 2, 2, 4, 4]], dtype=np.int32)
    assert np.all(blocks == blocks_2d(matrix))

    # Test row and column matrices.
    matrix = np.asarray([[
        .5, .1, .1, .1, .5, .5, 3, -3, 4, 5, 6, 6, 6, 7, 8]], dtype=np.float32)
    blocks = np.asarray([[
        0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9]], dtype=np.int32)
    assert np.all(blocks == blocks_2d(matrix))
    # Also with transpose.
    assert np.all(blocks.T == blocks_2d(matrix.T))

    # TODO(josipd): Try with non-2d, check that an exception is thrown.


def test_blockwise_means():
    blocks = np.asarray([
        0, 1, 0, 0, 1, 1, 0, 2], dtype=np.int32)
    vector = np.asarray([
        0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    b0 = np.mean([0, 2, 3, 6])
    b1 = np.mean([1, 4, 5])
    b2 = np.mean([7])
    expected = np.asarray([
        b0, b1, b0, b0, b1, b1, b0, b2])
    assert np.allclose(expected, blockwise_means(blocks, vector))
    assert np.allclose(expected, blockwise_means(blocks + 10, vector))
