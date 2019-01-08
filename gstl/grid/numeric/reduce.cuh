#pragma once

#ifndef GPU_GRID_NUMERIC_REDUCE_HPP
#define GPU_GRID_NUMERIC_REDUCE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class RandomOutputIt>
	GPU_DEVICE void reduce(grid_t g, RandomIt first, RandomIt last, RandomOutputIt buffer);

	template <class RandomIt, class RandomOutputIt, typename T>
	GPU_DEVICE void reduce(grid_t g, RandomIt first, RandomIt last, RandomOutputIt buffer, T init);

	template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp>
	GPU_DEVICE void reduce(grid_t g, RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op);
}

#include <gstl/grid/numeric/reduce.cu>

#endif // GPU_GRID_NUMERIC_REDUCE_HPP
