#pragma once

#ifndef GPU_ALGORITHMS_FIND_HPP
#define GPU_ALGORITHMS_FIND_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE RandomIt find(block_t g, RandomIt first, RandomIt last, const T& value);

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE RandomIt find(BlockTile g, RandomIt first, RandomIt last, const T& value);

	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR InputIt find(InputIt first, InputIt last, const T& value);

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE RandomIt find_if(block_t g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE RandomIt find_if(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR InputIt find_if(InputIt first, InputIt last, UnaryPredicate p);

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE RandomIt find_if_not(block_t g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE RandomIt find_if_not(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR InputIt find_if_not(InputIt first, InputIt last, UnaryPredicate p);

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool all_of(block_t g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool all_of(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool all_of(InputIt first, InputIt last, UnaryPredicate p);

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool any_of(block_t g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool any_of(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool any_of(InputIt first, InputIt last, UnaryPredicate p);

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool none_of(block_t g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool none_of(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool none_of(InputIt first, InputIt last, UnaryPredicate p);
}

#include <gstl/algorithms/find.cu>

#endif // GPU_ALGORITHMS_FIND_HPP
