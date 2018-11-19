#pragma once

#ifndef GPU_ALGORITHMS_COUNT_HPP
#define GPU_ALGORITHMS_COUNT_HPP

#include <gstl/utility/group_result.cuh>
#include <gstl/utility/iterator.cuh>

namespace gpu
{
	template <class It>
	using count_return_type = group_result<typename std::iterator_traits<It>::difference_type>;

	template <class RandomIt, typename T>
	GPU_DEVICE count_return_type<RandomIt> count(block_t g, RandomIt first, RandomIt last, const T& value);

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE count_return_type<RandomIt> count(BlockTile g, RandomIt first, RandomIt last, const T& value);

	template <class InputIt, typename T>
	GPU_DEVICE constexpr count_return_type<InputIt> count(InputIt first, InputIt last, const T& value);

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE count_return_type<RandomIt> count_if(block_t g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE count_return_type<RandomIt> count_if(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p);

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE constexpr count_return_type<InputIt> count_if(InputIt first, InputIt last, UnaryPredicate p);
}

#include <gstl/algorithms/count.cu>

#endif // GPU_ALGORITHMS_COUNT_HPP
