#pragma once

#ifndef GPU_NUMERIC_ADJACENT_DIFFERENCE_HPP
#define GPU_NUMERIC_ADJACENT_DIFFERENCE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomInputIt, class RandomOutputIt>
	GPU_DEVICE RandomOutputIt adjacent_difference(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first);

	template <class BlockTile, class RandomInputIt, class RandomOutputIt>
	GPU_DEVICE RandomOutputIt adjacent_difference(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first);

	template <class InputIt, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt adjacent_difference(InputIt first, InputIt last, OutputIt d_first);

	template <class RandomInputIt, class RandomOutputIt, class BinaryOperation>
	GPU_DEVICE RandomOutputIt adjacent_difference(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation op);

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, class BinaryOperation>
	GPU_DEVICE RandomOutputIt adjacent_difference(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation op);

	template< class InputIt, class OutputIt, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt adjacent_difference(InputIt first, InputIt last, OutputIt d_first, BinaryOperation op);
}

#include <gstl/numeric/adjacent_difference.cu>

#endif // GPU_NUMERIC_ADJACENT_DIFFERENCE_HPP
