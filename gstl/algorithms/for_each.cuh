#pragma once

#ifndef GPU_ALGORITHMS_FOR_EACH_HPP
#define GPU_ALGORITHMS_FOR_EACH_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class UnaryFunction>
	GPU_DEVICE void for_each(block_t g, RandomIt first, RandomIt last, UnaryFunction unary_op);

	template <class BlockTile, class RandomIt, class UnaryFunction>
	GPU_DEVICE void for_each(BlockTile g, RandomIt first, RandomIt last, UnaryFunction unary_op);

	template <class ForwardIt, class UnaryFunction>
	GPU_DEVICE GPU_CONSTEXPR void for_each(ForwardIt first, ForwardIt last, UnaryFunction unary_op);

	template <class ForwardIt, class Size, class UnaryFunction>
	GPU_DEVICE ForwardIt for_each_n(block_t g, ForwardIt first, Size n, UnaryFunction unary_op);

	template <class BlockTile, class ForwardIt, class Size, class UnaryFunction>
	GPU_DEVICE ForwardIt for_each_n(BlockTile g, ForwardIt first, Size n, UnaryFunction unary_op);

	template <class InputIt, class Size, class UnaryFunction>
	GPU_DEVICE GPU_CONSTEXPR InputIt for_each_n(InputIt first, Size n, UnaryFunction unary_op);
}

#include <gstl/algorithms/for_each.cu>

#endif // GPU_ALGORITHMS_FOR_EACH_HPP
