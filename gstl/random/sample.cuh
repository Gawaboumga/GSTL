#pragma once

#ifndef GPU_RANDOM_SAMPLE_HPP
#define GPU_RANDOM_SAMPLE_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/algorithms/generate.cuh>
#include <gstl/random/linear_congruential_engine.cuh>
#include <gstl/random/random_device.cuh>
#include <gstl/random/uniform_int_distribution.cuh>
#include <gstl/utility/limits.cuh>

namespace gpu
{
	template <class OutputIt, class Size>
	GPU_DEVICE void randint(block_t g, OutputIt d_first, Size size);

	template <class BlockTile, class OutputIt, class Size>
	GPU_DEVICE void randint(BlockTile g, OutputIt d_first, Size size);

	template <class OutputIt, class Size, class Integral>
	GPU_DEVICE void randint(block_t g, OutputIt d_first, Size size, Integral low);

	template <class BlockTile, class OutputIt, class Size, class Integral>
	GPU_DEVICE void randint(BlockTile g, OutputIt d_first, Size size, Integral low);

	template <class OutputIt, class Size, class Integral>
	GPU_DEVICE void randint(block_t g, OutputIt d_first, Size size, Integral low, Integral high);

	template <class BlockTile, class OutputIt, class Size, class Integral>
	GPU_DEVICE void randint(BlockTile g, OutputIt d_first, Size size, Integral low, Integral high);
}

#include <gstl/random/sample.cu>

#endif // GPU_RANDOM_SAMPLE_HPP
