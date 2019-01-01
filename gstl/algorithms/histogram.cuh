#pragma once

#ifndef GPU_ALGORITHMS_HISTOGRAM_HPP
#define GPU_ALGORITHMS_HISTOGRAM_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class OutputIt, class MappingFunction>
	GPU_DEVICE void histogram(block_t g, RandomIt first, RandomIt last, OutputIt d_first, MappingFunction unary_op);

	template <class BlockTile, class RandomIt, class OutputIt, class MappingFunction>
	GPU_DEVICE void histogram(BlockTile g, RandomIt first, RandomIt last, OutputIt d_first, MappingFunction unary_op);

	template <class Input, class OutputIt, class MappingFunction>
	GPU_DEVICE GPU_CONSTEXPR void histogram(Input first, Input last, OutputIt d_first, MappingFunction unary_op);
}

#include <gstl/algorithms/histogram.cu>

#endif // GPU_ALGORITHMS_HISTOGRAM_HPP
