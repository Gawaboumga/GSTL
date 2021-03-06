#pragma once

#ifndef GPU_ALGORITHMS_ENUMERATE_HPP
#define GPU_ALGORITHMS_ENUMERATE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class Function>
	GPU_DEVICE void enumerate(block_t g, RandomIt first, RandomIt last, Function f);

	template <class BlockTile, class RandomIt, class Function>
	GPU_DEVICE void enumerate(BlockTile g, RandomIt first, RandomIt last, Function f);

	template <class ForwardIt, class Function>
	GPU_DEVICE GPU_CONSTEXPR void enumerate(ForwardIt first, ForwardIt last, Function f);
}

#include <gstl/algorithms/enumerate.cu>

#endif // GPU_ALGORITHMS_ENUMERATE_HPP
