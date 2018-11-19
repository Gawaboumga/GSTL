#pragma once

#ifndef GPU_UTILITY_SHFL_HPP
#define GPU_UTILITY_SHFL_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <typename T>
	GPU_DEVICE T shfl(block_t g, T value, unsigned int thid = 0u);

	template <class BlockTile, typename T>
	GPU_DEVICE T shfl(BlockTile g, T value, unsigned int thid = 0u);
}

#include <gstl/utility/shfl.cu>

#endif // GPU_UTILITY_SHFL_HPP
