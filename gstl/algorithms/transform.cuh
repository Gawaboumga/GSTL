#pragma once

#ifndef GPU_ALGORITHMS_TRANSFORM_HPP
#define GPU_ALGORITHMS_TRANSFORM_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt1, class RandomIt2, class UnaryOperation>
	GPU_DEVICE RandomIt2 transform(block_t g, RandomIt1 first, RandomIt1 last, RandomIt2 d_first, UnaryOperation unary_op);

	template <class RandomIt1, class RandomIt2, class UnaryOperation, int tile_size>
	GPU_DEVICE RandomIt2 transform(block_tile_t<tile_size> g, RandomIt1 first, RandomIt1 last, RandomIt2 d_first, UnaryOperation unary_op);

	template <class ForwardIt, class OutputIt, class UnaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform(ForwardIt first, ForwardIt last, OutputIt d_first, UnaryOperation unary_op);

	template <class RandomIt1, class RandomIt2, class RandomIt3, class BinaryOperation>
	GPU_DEVICE RandomIt3 transform(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt3 d_first, BinaryOperation binary_op);

	template <class RandomIt1, class RandomIt2, class RandomIt3, class BinaryOperation, int tile_size>
	GPU_DEVICE RandomIt3 transform(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt3 d_first, BinaryOperation binary_op);

	template <class ForwardIt1, class ForwardIt2, class OutputIt, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, OutputIt d_first, BinaryOperation binary_op);
}

#include <gstl/algorithms/transform.cu>

#endif // GPU_ALGORITHMS_TRANSFORM_HPP
