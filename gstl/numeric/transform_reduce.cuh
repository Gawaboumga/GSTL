#pragma once

#ifndef GPU_NUMERIC_TRANSFORM_REDUCE_HPP
#define GPU_NUMERIC_TRANSFORM_REDUCE_HPP

#include <gstl/prerequisites.hpp>
#include <gstl/utility/group_result.cuh>

namespace gpu
{
	template <class RandomIt1, class RandomIt2, typename T>
	GPU_DEVICE group_result<T> transform_reduce(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T init);

	template <class BlockTile, class RandomIt1, class RandomIt2, typename T>
	GPU_DEVICE group_result<T> transform_reduce(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T init);

	template <class InputIt1, class InputIt2, typename T>
	GPU_DEVICE GPU_CONSTEXPR T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init);

	template <class RandomIt1, class RandomIt2, typename T, class BinaryOp1, class BinaryOp2>
	GPU_DEVICE group_result<T> transform_reduce(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2);

	template <class BlockTile, class RandomIt1, class RandomIt2, typename T, class BinaryOp1, class BinaryOp2>
	GPU_DEVICE group_result<T> transform_reduce(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2);

	template <class InputIt1, class InputIt2, typename T, class BinaryOp1, class BinaryOp2>
	GPU_DEVICE GPU_CONSTEXPR T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2);

	template <class RandomIt, typename T, class BinaryOp, class UnaryOp>
	GPU_DEVICE group_result<T> transform_reduce(block_t g, RandomIt first, RandomIt last, T init, BinaryOp binary_op, UnaryOp unary_op);

	template <class BlockTile, class RandomIt, typename T, class BinaryOp, class UnaryOp>
	GPU_DEVICE group_result<T> transform_reduce(BlockTile g, RandomIt first, RandomIt last, T init, BinaryOp binary_op, UnaryOp unary_op);

	template <class InputIt, typename T, class BinaryOp, class UnaryOp>
	GPU_DEVICE GPU_CONSTEXPR T transform_reduce(InputIt first, InputIt last, T init, BinaryOp binary_op, UnaryOp unary_op);
}

#include <gstl/numeric/transform_reduce.cu>

#endif // GPU_NUMERIC_TRANSFORM_REDUCE_HPP
