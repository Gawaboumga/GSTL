#pragma once

#ifndef GPU_ALGORITHMS_INPLACE_MERGE_HPP
#define GPU_ALGORITHMS_INPLACE_MERGE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class BidirIt>
	GPU_DEVICE void inplace_merge(block_t g, BidirIt first, BidirIt middle, BidirIt last);

	template <class BlockTile, class BidirIt>
	GPU_DEVICE void inplace_merge(BlockTile g, BidirIt first, BidirIt middle, BidirIt last);

	template <class BidirIt>
	GPU_DEVICE GPU_CONSTEXPR void inplace_merge(BidirIt first, BidirIt middle, BidirIt last);

	template <class BidirIt, class Compare>
	GPU_DEVICE void inplace_merge(block_t g, BidirIt first, BidirIt middle, BidirIt last, Compare comp);

	template <class BlockTile, class BidirIt, class Compare>
	GPU_DEVICE void inplace_merge(BlockTile g, BidirIt first, BidirIt middle, BidirIt last, Compare comp);

	template <class BidirIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR void inplace_merge(BidirIt first, BidirIt middle, BidirIt last, Compare comp);
}

#include <gstl/algorithms/inplace_merge.cu>

#endif // GPU_ALGORITHMS_INPLACE_MERGE_HPP
