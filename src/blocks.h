#pragma once

#include <pybind11/numpy.h>


pybind11::array_t<float> blockwise_means(const pybind11::array_t<int>& blocks,
                                         const pybind11::array_t<float>& input);
pybind11::array_t<int> blocks_2d(const pybind11::array_t<float>& matrix);
