#pragma once

#ifndef GPU_ALGORITHMS_MOVE_HPP
#define GPU_ALGORITHMS_MOVE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class ForwardIt>
	GPU_DEVICE ForwardIt move(block_t g, RandomIt first, RandomIt last, ForwardIt d_first);

	template <class BlockTile, class RandomIt, class ForwardIt>
	GPU_DEVICE ForwardIt move(BlockTile g, RandomIt first, RandomIt last, ForwardIt d_first);

	template <class InputIt, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt move(InputIt first, InputIt last, OutputIt d_first);

	template <class RandomIt, class BidirIt>
	GPU_DEVICE BidirIt move_backward(block_t g, RandomIt first, RandomIt last, BidirIt d_last);

	template <class BlockTile, class RandomIt, class BidirIt>
	GPU_DEVICE BidirIt move_backward(BlockTile g, RandomIt first, RandomIt last, BidirIt d_last);

	template <class BidirIt1, class BidirIt2>
	GPU_DEVICE GPU_CONSTEXPR BidirIt2 move_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last);
}

#include <gstl/algorithms/move.cu>

#endif // GPU_ALGORITHMS_MOVE_HPP
