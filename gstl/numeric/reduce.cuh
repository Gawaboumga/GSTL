#pragma once

#ifndef GPU_NUMERIC_REDUCE_HPP
#define GPU_NUMERIC_REDUCE_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/utility/group_result.cuh>

#include <iterator>

namespace gpu
{
	template <class iterator>
	using reduce_return_type = typename std::iterator_traits<iterator>::value_type;

	template <class RandomIt>
	GPU_DEVICE group_result<reduce_return_type<RandomIt>> reduce(block_t g, RandomIt first, RandomIt last);

	template <class BlockTile, class RandomIt>
	GPU_DEVICE group_result<reduce_return_type<RandomIt>> reduce(BlockTile g, RandomIt first, RandomIt last);

	template <class InputIt>
	GPU_DEVICE reduce_return_type<InputIt> reduce(InputIt first, InputIt last);

	template <class RandomIt, typename T>
	GPU_DEVICE group_result<T> reduce(block_t g, RandomIt first, RandomIt last, T init);

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE group_result<T> reduce(BlockTile g, RandomIt first, RandomIt last, T init);

	template <class InputIt, typename T>
	GPU_DEVICE T reduce(InputIt first, InputIt last, T init);

	template <class RandomIt, typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(block_t g, RandomIt first, RandomIt last, T init, BinaryOp binary_op);

	template <class BlockTile, class RandomIt, typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(BlockTile g, RandomIt first, RandomIt last, T init, BinaryOp binary_op);

	template <class InputIt, typename T, class BinaryOp>
	GPU_DEVICE T reduce(InputIt first, InputIt last, T init, BinaryOp binary_op);

	template <typename T>
	GPU_DEVICE group_result<T> reduce(block_t g, T value);

	template <class BlockTile, typename T>
	GPU_DEVICE group_result<T> reduce(block_t g, T value, unsigned int maximal_lane);

	template <class BlockTile, typename T>
	GPU_DEVICE group_result<T> reduce(BlockTile g, T value);

	template <class BlockTile, typename T>
	GPU_DEVICE group_result<T> reduce(BlockTile g, T value, unsigned int maximal_lane);

	template <typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(block_t g, T value, BinaryOp binary_op);

	template <typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(block_t g, T value, BinaryOp binary_op, unsigned int maximal_lane);

	template <class BlockTile, typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(BlockTile g, T value, BinaryOp binary_op);

	template <class BlockTile, typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(BlockTile g, T value, BinaryOp binary_op, unsigned int maximal_lane);
}

#include <gstl/numeric/reduce.cu>

#endif // GPU_NUMERIC_REDUCE_HPP
