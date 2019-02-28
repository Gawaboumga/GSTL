#pragma once

#ifndef GPU_ALGORITHMS_SWAP_HPP
#define GPU_ALGORITHMS_SWAP_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt1, class RandomIt2>
	GPU_DEVICE RandomIt2 swap_ranges(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2);

	template <class BlockTile, class RandomIt1, class RandomIt2>
	GPU_DEVICE RandomIt2 swap_ranges(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2);

	template <class ForwardIt1, class ForwardIt2>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt2 swap_ranges(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);

	template <class RandomIt1, class RandomIt2, class Size>
	GPU_DEVICE RandomIt2 swap_ranges_n(block_t g, RandomIt1 first1, RandomIt2 first2, Size n);

	template <class BlockTile, class RandomIt1, class RandomIt2, class Size>
	GPU_DEVICE RandomIt2 swap_ranges_n(BlockTile g, RandomIt1 first1, RandomIt2 first2, Size n);

	template <class ForwardIt1, class ForwardIt2, class Size>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt2 swap_ranges_n(ForwardIt1 first1, ForwardIt2 first2, Size n);
}

#include <gstl/algorithms/swap.cu>

#endif // GPU_ALGORITHMS_SWAP_HPP
