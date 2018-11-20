#pragma once

#ifndef GPU_ALGORITHMS_ADJACENT_FIND_HPP
#define GPU_ALGORITHMS_ADJACENT_FIND_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class ForwardIt>
	GPU_DEVICE ForwardIt adjacent_find(block_t g, ForwardIt first, ForwardIt last);

	template <class BlockTile, class ForwardIt>
	GPU_DEVICE ForwardIt adjacent_find(BlockTile g, ForwardIt first, ForwardIt last);

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt adjacent_find(ForwardIt first, ForwardIt last);

	template <class ForwardIt, class BinaryPredicate>
	GPU_DEVICE ForwardIt adjacent_find(block_t g, ForwardIt first, ForwardIt last, BinaryPredicate p);

	template <class BlockTile, class ForwardIt, class BinaryPredicate>
	GPU_DEVICE ForwardIt adjacent_find(BlockTile g, ForwardIt first, ForwardIt last, BinaryPredicate p);

	template <class ForwardIt, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt adjacent_find(ForwardIt first, ForwardIt last, BinaryPredicate p);
}

#include <gstl/algorithms/adjacent_find.cu>

#endif // GPU_ALGORITHMS_ADJACENT_FIND_HPP
