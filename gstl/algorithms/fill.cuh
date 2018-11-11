#pragma once

#ifndef GPU_ALGORITHMS_FILL_HPP
#define GPU_ALGORITHMS_FILL_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE void fill(block_t g, RandomIt first, RandomIt last, const T& value);

	template <class RandomIt, typename T, int tile_size>
	GPU_DEVICE void fill(block_tile_t<tile_size> g, RandomIt first, RandomIt last, const T& value);

	template <class ForwardIt, class T>
	GPU_DEVICE GPU_CONSTEXPR void fill(ForwardIt first, ForwardIt last, const T& value);
}

#include <gstl/algorithms/fill.cu>

#endif // GPU_ALGORITHMS_FILL_HPP
