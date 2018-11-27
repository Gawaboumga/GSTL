#pragma once

#ifndef GPU_ALGORITHMS_BINARY_SEARCH_HPP
#define GPU_ALGORITHMS_BINARY_SEARCH_HPP

#include <gstl/prerequisites.hpp>
#include <gstl/utility/pair.cuh>

namespace gpu
{
	template <class ForwardIt, typename T>
	GPU_DEVICE bool binary_search(block_t g, ForwardIt first, ForwardIt last, const T& value);

	template <class BlockTile, class ForwardIt, typename T>
	GPU_DEVICE bool binary_search(BlockTile g, ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR bool binary_search(ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE bool binary_search(block_t g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class BlockTile, class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE bool binary_search(BlockTile g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool binary_search(ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class ForwardIt, typename T>
	GPU_DEVICE pair<ForwardIt, ForwardIt> equal_range(block_t g, ForwardIt first, ForwardIt last, const T& value);

	template <class BlockTile, class ForwardIt, typename T>
	GPU_DEVICE pair<ForwardIt, ForwardIt> equal_range(BlockTile g, ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR pair<ForwardIt, ForwardIt> equal_range(ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE pair<ForwardIt, ForwardIt> equal_range(block_t g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class BlockTile, class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE pair<ForwardIt, ForwardIt> equal_range(BlockTile g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR pair<ForwardIt, ForwardIt> equal_range(ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class ForwardIt, typename T>
	GPU_DEVICE ForwardIt lower_bound(block_t g, ForwardIt first, ForwardIt last, const T& value);

	template <class BlockTile, class ForwardIt, typename T>
	GPU_DEVICE ForwardIt lower_bound(BlockTile g, ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE ForwardIt lower_bound(block_t g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class BlockTile, class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE ForwardIt lower_bound(BlockTile g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class ForwardIt, typename T>
	GPU_DEVICE ForwardIt upper_bound(block_t g, ForwardIt first, ForwardIt last, const T& value);

	template <class BlockTile, class ForwardIt, typename T>
	GPU_DEVICE ForwardIt upper_bound(BlockTile g, ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE ForwardIt upper_bound(block_t g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class BlockTile, class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE ForwardIt upper_bound(BlockTile g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);
}

#include <gstl/algorithms/binary_search.cu>

#endif // GPU_ALGORITHMS_BINARY_SEARCH_HPP
