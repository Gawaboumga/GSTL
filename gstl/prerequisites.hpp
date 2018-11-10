#ifndef GPU_PREREQUISITES_HPP
#define GPU_PREREQUISITES_HPP

#include <cooperative_groups.h>

#include <assert.hpp>
#include <debug_configuration.hpp>

#include <cstdint>

namespace gpu
{
	using I32 = std::int32_t;
	using I64 = std::int64_t;

	using UI32 = std::uint32_t;
	using UI64 = std::uint64_t;

	using ptrdiff_t = I32;
	using size_t = I32;

	using Block = cooperative_groups::thread_block;
	template <int tile_sz>
	using Warp = cooperative_groups::thread_block_tile<tile_sz>;

	static constexpr I32 MAX_NUMBER_OF_THREADS_PER_WARP = 32u;
	static constexpr I32 MAX_NUMBER_OF_WARPS_PER_BLOCK = 32u;
	static constexpr I32 MAX_NUMBER_OF_THREADS_PER_BLOCK = MAX_NUMBER_OF_WARPS_PER_BLOCK * MAX_NUMBER_OF_THREADS_PER_WARP;

	#define GPU_CONSTEXPR constexpr
	#define GPU_DEVICE __device__
	#define GPU_HOST __host__
}

#endif // GPU_PREREQUISITES_HPP
