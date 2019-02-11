#pragma once

#ifndef GPU_ALGORITHMS_SWAP_HPP
#define GPU_ALGORITHMS_SWAP_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class ForwardIt1, class ForwardIt2>
	GPU_DEVICE ForwardIt2 swap_ranges(block_t g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);

	template <class BlockTile, class ForwardIt1, class RandomIt2>
	GPU_DEVICE ForwardIt2 swap_ranges(BlockTile g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);

	template <class ForwardIt1, class ForwardIt2>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt2 swap_ranges(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);
}

#include <gstl/algorithms/swap.cu>

#endif // GPU_ALGORITHMS_SWAP_HPP
