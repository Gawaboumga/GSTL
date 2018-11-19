#pragma once

#ifndef GPU_NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_HPP
#define GPU_NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation>
	GPU_DEVICE RandomOutputIt transform_exclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op);

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation>
	GPU_DEVICE RandomOutputIt transform_exclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op);

	template <class InputIt, class OutputIt, class BinaryOperation, class UnaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform_exclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op);

	template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
	GPU_DEVICE RandomOutputIt transform_exclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init);

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
	GPU_DEVICE RandomOutputIt transform_exclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init);

	template <class InputIt, class OutputIt, class BinaryOperation, class UnaryOperation, typename T>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform_exclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init);

	template <class BinaryOperation, typename T>
	GPU_DEVICE T transform_exclusive_scan(block_t g, T value, BinaryOperation binary_op, T init);

	template <class BlockTile, class BinaryOperation, typename T>
	GPU_DEVICE T transform_exclusive_scan(BlockTile g, T value, BinaryOperation binary_op, T init);
}

#include <gstl/numeric/transform_exclusive_scan.cu>

#endif // GPU_NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_HPP
