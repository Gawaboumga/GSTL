#pragma once

#ifndef GPU_ALGORITHMS_FILL_HPP
#define GPU_ALGORITHMS_FILL_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE void fill(block_t g, RandomIt first, RandomIt last, const T& value);

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE void fill(BlockTile g, RandomIt first, RandomIt last, const T& value);

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR void fill(ForwardIt first, ForwardIt last, const T& value);

	template <class RandomIt, class Size, typename T>
	GPU_DEVICE void fill_n(block_t g, RandomIt first, Size n, const T& value);

	template <class BlockTile, class RandomIt, class Size, typename T>
	GPU_DEVICE void fill_n(BlockTile g, RandomIt first, Size n, const T& value);

	template <class OutputIt, class Size, typename T>
	GPU_DEVICE GPU_CONSTEXPR OutputIt fill_n(OutputIt first, Size n, const T& value);
}

#include <gstl/algorithms/fill.cu>

#endif // GPU_ALGORITHMS_FILL_HPP
