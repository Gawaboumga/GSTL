#pragma once

#ifndef GPU_NUMERIC_EXCLUSIVESCAN_HPP
#define GPU_NUMERIC_EXCLUSIVESCAN_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomInputIt, class RandomOutputIt, typename T>
	GPU_DEVICE RandomOutputIt exclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, T init);

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, typename T>
	GPU_DEVICE RandomOutputIt exclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, T init);

	template <class InputIt, class OutputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init);

	template <class RandomInputIt, class RandomOutputIt, typename T, class BinaryOperation>
	GPU_DEVICE RandomOutputIt exclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, T init, BinaryOperation binary_op);

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, typename T, class BinaryOperation>
	GPU_DEVICE RandomOutputIt exclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, T init, BinaryOperation binary_op);

	template <class InputIt, class OutputIt, typename T, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init, BinaryOperation binary_op);

	template <class BlockTile, typename T>
	GPU_DEVICE T exclusive_scan(BlockTile g, T value, T init);

	template <class BlockTile, typename T, class BinaryOperation>
	GPU_DEVICE T exclusive_scan(BlockTile g, T value, T init, BinaryOperation binary_op);
}

#include <gstl/numeric/exclusive_scan.cu>

#endif // GPU_NUMERIC_EXCLUSIVESCAN_HPP
