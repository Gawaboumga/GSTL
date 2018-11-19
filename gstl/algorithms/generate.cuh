#pragma once

#ifndef GPU_ALGORITHMS_GENERATE_HPP
#define GPU_ALGORITHMS_GENERATE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class Generator>
	GPU_DEVICE void generate(block_t block, RandomIt first, RandomIt last, Generator g);

	template <class BlockTile, class RandomIt, class Generator>
	GPU_DEVICE void generate(BlockTile warp, RandomIt first, RandomIt last, Generator g);

	template <class ForwardIt, class Generator>
	GPU_DEVICE GPU_CONSTEXPR void generate(ForwardIt first, ForwardIt last, Generator g);
}

#include <gstl/algorithms/generate.cu>

#endif // GPU_ALGORITHMS_GENERATE_HPP
