#pragma once

#ifndef GPU_ALGORITHMS_COPY_HPP
#define GPU_ALGORITHMS_COPY_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class ForwardIt>
	GPU_DEVICE ForwardIt copy(block_t g, RandomIt first, RandomIt last, ForwardIt d_first);

	template <class BlockTile, class RandomIt, class ForwardIt>
	GPU_DEVICE ForwardIt copy(BlockTile g, RandomIt first, RandomIt last, ForwardIt d_first);

	template <class InputIt, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt copy(InputIt first, InputIt last, OutputIt d_first);

	template <class RandomIt, class ForwardIt, class UnaryPredicate>
	GPU_DEVICE ForwardIt copy_if(block_t g, RandomIt first, RandomIt last, ForwardIt d_first, UnaryPredicate p);

	template <class BlockTile, class RandomIt, class ForwardIt, class UnaryPredicate>
	GPU_DEVICE ForwardIt copy_if(BlockTile g, RandomIt first, RandomIt last, ForwardIt d_first, UnaryPredicate p);

	template <class InputIt, class OutputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR OutputIt copy_if(InputIt first, InputIt last, OutputIt d_first, UnaryPredicate p);

	template <class RandomIt, class Size, class ForwardIt>
	GPU_DEVICE ForwardIt copy_n(block_t g, RandomIt first, Size count, ForwardIt d_first);

	template <class BlockTile, class RandomIt, class Size, class ForwardIt>
	GPU_DEVICE ForwardIt copy_n(BlockTile g, RandomIt first, Size count, ForwardIt d_first);

	template <class InputIt, class Size, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt copy_n(InputIt first, Size count, OutputIt d_first);

	template <class BidirIt1, class BidirIt2>
	GPU_DEVICE GPU_CONSTEXPR BidirIt2 copy_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last);
}

#include <gstl/algorithms/copy.cu>

#endif // GPU_ALGORITHMS_COPY_HPP
