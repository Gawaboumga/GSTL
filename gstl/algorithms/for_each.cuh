#pragma once

#ifndef GPU_ALGORITHMS_FOR_EACH_HPP
#define GPU_ALGORITHMS_FOR_EACH_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class UnaryFunction>
	GPU_DEVICE void for_each(block_t g, RandomIt first, RandomIt last, UnaryFunction unary_op);

	template <class RandomIt, class UnaryFunction, int tile_size>
	GPU_DEVICE void for_each(block_tile_t<tile_size> g, RandomIt first, RandomIt last, UnaryFunction unary_op);

	template <class ForwardIt, class UnaryFunction>
	GPU_DEVICE GPU_CONSTEXPR void for_each(ForwardIt first, ForwardIt last, UnaryFunction unary_op);
}

#include <gstl/algorithms/for_each.cu>

#endif // GPU_ALGORITHMS_FOR_EACH_HPP
