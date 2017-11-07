#include "blocks.h"

#include <algorithm>
#include <string>
#include <vector>
#include <boost/pending/disjoint_sets.hpp>

namespace py = pybind11;

namespace {

const char* blockwise_means_doc = R"(blockwise_means(blocks, input)

Average the elements of the given vector within each block.

Specifically, the coordinate ``i`` of the returned vector will contain the mean
of all entries ``j`` in ``input`` that have ``blocks[j]=b``.

Arguments
---------
blocks : numpy.ndarray
    A vector of ints that denote the block memberships of each position.

    The set of numbers in this vector should start at zero and be consecutive.

input : numpy.ndarray
    A vector of same size as ``block`` containing the data to be averaged.

Returns
--------
numpy.ndarray
    A vector of same size as the inputs that contains the block-wise averages.
)";

}  // namespace

py::array_t<float> blockwise_means(const py::array_t<int>& blocks_,
                                   const py::array_t<float>& input_) {
    if (blocks_.ndim() != 1 || input_.ndim() != 1) {
        throw std::runtime_error("the given arrays must be one dimensional");
    }
    if (blocks_.shape(0) != input_.shape(0)) {
        throw std::runtime_error("the number of elements in the must match");
    }

    auto blocks = blocks_.unchecked<1>();
    auto input = input_.unchecked<1>();

    int n_blocks = 0;
    for (int i = 0; i < blocks.shape(0); i++) {
        if (blocks[i] < 0) {
            throw std::runtime_error("the block ids must be non-negative");
        }
        n_blocks = std::max(n_blocks, 1 + blocks(i));
    }

    std::vector<float> blocks_sums(n_blocks);
    std::vector<int> blocks_total(n_blocks);

    for (int i = 0; i < blocks.shape(0); i++) {
        blocks_sums[blocks(i)] += input[i];
        ++blocks_total[blocks[i]];
    }
    for (int i = 0; i < n_blocks; i++) {
        blocks_sums[i] /= static_cast<float>(blocks_total[i]);
    }

    py::array_t<float> output_ = py::array_t<float>(blocks.shape(0));
    auto output = output_.mutable_unchecked<1>();
    for (int i = 0; i < output_.shape(0); i++) {
        output(i) = blocks_sums[blocks[i]];
    }
    return output_;
}


namespace {

const char* blocks_2d_doc = R"(blocks_2d(matrix)

Return the connected components of the matrix.

Two positions are connected iff they hold the same value, and they differ in
one coordinate by one (i.e., it is a 4-connected grid).

Arguments
---------
matrix : numpy.ndarray
    The two-dimensional matrix.

Returns
--------
numpy.ndarray
    A matrix of ints, same size as the input.

    The positions corresponding to the same connected component have the
    same label. The labels are consecutive integers starting at zero.
)";

}  // namespace


py::array_t<int> blocks_2d(const py::array_t<float>& matrix_) {
    if (matrix_.ndim() != 2) {
        throw std::runtime_error("the given matrix must be two dimensional");
    }
    auto matrix = matrix_.unchecked<2>();

    std::vector<int> ranks(matrix.size());
    std::vector<int> parents(matrix.size());
    boost::disjoint_sets<int*, int*> union_find(&ranks[0], &parents[0]);

    #define IDX(i, j) ((i) * static_cast<int>(matrix.shape(1)) + (j))

    for (int i = 0; i < matrix_.shape(0); i++) {
        for (int j = 0; j < matrix_.shape(1); j++) {
            int idx = IDX(i, j);
            union_find.make_set(idx);
            if (i > 0 && matrix(i, j) == matrix(i - 1, j)) {
                union_find.union_set(idx, IDX(i - 1, j));
            }
            if (j > 0 && matrix(i, j) == matrix(i, j - 1)) {
                union_find.union_set(idx, IDX(i, j - 1));
            }
        }
    }

    std::unordered_map<int, int> root_to_idx;
    py::array_t<int> output_ = py::array_t<int>({matrix_.shape(0),
                                                 matrix_.shape(1)});
    auto output = output_.mutable_unchecked<2>();
    int next_id = 0;
    for (int i = 0; i < matrix.shape(0); i++) {
        for (int j = 0; j < matrix.shape(1); j++) {
            int idx = IDX(i, j);
            int root = union_find.find_set(idx);
            auto iter = root_to_idx.find(root);
            if (iter == root_to_idx.end()) {
                output(i, j) = next_id;
                root_to_idx[root] = next_id++;
            } else {
                output(i, j) = iter->second;
            }
        }
    }

    return output_;
}


PYBIND11_MODULE(blocks, m) {
    py::options options;
    options.disable_function_signatures();
    m.def("blockwise_means", blockwise_means, blockwise_means_doc);
    m.def("blocks_2d", blocks_2d, blocks_2d_doc);
}
