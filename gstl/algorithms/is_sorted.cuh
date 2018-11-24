#pragma once

#ifndef GPU_ALGORITHMS_IS_SORTED_HPP
#define GPU_ALGORITHMS_IS_SORTED_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class ForwardIt>
	GPU_DEVICE bool is_sorted(block_t g, ForwardIt first, ForwardIt last);

	template <class BlockTile, class ForwardIt>
	GPU_DEVICE bool is_sorted(BlockTile g, ForwardIt first, ForwardIt last);

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted(ForwardIt first, ForwardIt last);

	template <class ForwardIt, class Compare>
	GPU_DEVICE bool is_sorted(block_t g, ForwardIt first, ForwardIt last, Compare comp);

	template <class BlockTile, class ForwardIt, class Compare>
	GPU_DEVICE bool is_sorted(BlockTile g, ForwardIt first, ForwardIt last, Compare comp);

	template <class ForwardIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted(ForwardIt first, ForwardIt last, Compare comp);

	template <class ForwardIt>
	GPU_DEVICE ForwardIt is_sorted_until(block_t g, ForwardIt first, ForwardIt last);

	template <class BlockTile, class ForwardIt>
	GPU_DEVICE ForwardIt is_sorted_until(BlockTile g, ForwardIt first, ForwardIt last);

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt is_sorted_until(ForwardIt first, ForwardIt last);

	template <class ForwardIt, class Compare>
	GPU_DEVICE ForwardIt is_sorted_until(block_t g, ForwardIt first, ForwardIt last, Compare comp);

	template <class BlockTile, class ForwardIt, class Compare>
	GPU_DEVICE ForwardIt is_sorted_until(BlockTile g, ForwardIt first, ForwardIt last, Compare comp);

	template <class ForwardIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt is_sorted_until(ForwardIt first, ForwardIt last, Compare comp);
}

#include <gstl/algorithms/is_sorted.cu>

#endif // GPU_ALGORITHMS_IS_SORTED_HPP
