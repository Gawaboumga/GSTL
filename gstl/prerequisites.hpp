#pragma once

#ifndef GPU_PREREQUISITES_HPP
#define GPU_PREREQUISITES_HPP

#define GPU_CONSTEXPR constexpr
#define GPU_DEVICE __device__
#define GPU_GLOBAL __global__
#define GPU_HOST __host__
#define GPU_SHARED __shared__

#include <gstl/debug_configuration.hpp>
#include <gstl/assert.cuh>

#include <cooperative_groups.h>

#include <cstdint>

namespace gpu
{
	namespace detail
	{
		struct grid_t
		{
			GPU_DEVICE inline dim3 grid_dim() const
			{
				return gridDim;
			}

			GPU_DEVICE inline unsigned int size() const
			{
				return blockDim.x * gridDim.x;
			}

			GPU_DEVICE inline unsigned int thread_rank() const
			{
				return blockIdx.x * blockDim.x + threadIdx.x;
			}
		};

		GPU_DEVICE inline grid_t this_grid()
		{
			return grid_t();
		}
	}

	using I32 = std::int32_t;
	using I64 = std::int64_t;

	using UI32 = std::uint32_t;
	using UI64 = std::uint64_t;

	using offset_t = I32;
	using ptrdiff_t = I32;
	using size_t = I32;
	using uintptr_t = std::uintptr_t;

	using grid_t = detail::grid_t;
	using block_t = cooperative_groups::thread_block;
	template <unsigned int tile_sz>
	using block_tile_t = cooperative_groups::thread_block_tile<tile_sz>;

	GPU_DEVICE inline grid_t this_grid()
	{
		return detail::this_grid();
	}

	GPU_DEVICE inline block_t this_thread_block()
	{
		return cooperative_groups::this_thread_block();
	}

	template <unsigned int tile_sz>
	GPU_DEVICE inline block_tile_t<tile_sz> tiled_partition(block_t block)
	{
		return cooperative_groups::tiled_partition<tile_sz>(block);
	}

	static constexpr I32 DEFAULT_BYTE_ALIGNMENT = 32u;
	static constexpr I32 MAX_NUMBER_OF_THREADS_PER_WARP = 32u;
	static constexpr I32 MAX_NUMBER_OF_WARPS_PER_BLOCK = 32u;
	static constexpr I32 MAX_NUMBER_OF_THREADS_PER_BLOCK = MAX_NUMBER_OF_WARPS_PER_BLOCK * MAX_NUMBER_OF_THREADS_PER_WARP;
}

#endif // GPU_PREREQUISITES_HPP
