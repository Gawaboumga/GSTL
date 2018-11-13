#include <gstl/utility/ballot.cuh>

#include <gstl/containers/array.cuh>

namespace gpu
{
	GPU_DEVICE bool all(block_t g, bool value)
	{
		GPU_SHARED gpu::array<bool, MAX_NUMBER_OF_WARPS_PER_BLOCK> results;

		offset_t warp_id = g.thread_rank() / MAX_NUMBER_OF_THREADS_PER_WARP;
		block_tile_t<MAX_NUMBER_OF_THREADS_PER_WARP> warp = tiled_partition<MAX_NUMBER_OF_THREADS_PER_WARP>(g);
		bool warp_result = all(warp, value);
		if (warp.thread_rank() == 0)
			results[warp_id] = warp_result;

		g.sync();
		bool warp_results = true;
		offset_t number_of_warps = g.size() / MAX_NUMBER_OF_THREADS_PER_WARP;
		offset_t local_warp_id = g.thread_rank() % MAX_NUMBER_OF_THREADS_PER_WARP;
		if (local_warp_id < number_of_warps)
			warp_results = results[local_warp_id];
		
		return all(warp, warp_results);
	}

	template <int tile_sz>
	GPU_DEVICE bool all(block_tile_t<tile_sz> g, bool value)
	{
		return g.all(value);
	}

	GPU_DEVICE bool any(block_t g, bool value)
	{
		GPU_SHARED gpu::array<bool, MAX_NUMBER_OF_WARPS_PER_BLOCK> results;

		offset_t warp_id = g.thread_rank() / MAX_NUMBER_OF_THREADS_PER_WARP;
		block_tile_t<MAX_NUMBER_OF_THREADS_PER_WARP> warp = tiled_partition<MAX_NUMBER_OF_THREADS_PER_WARP>(g);
		bool warp_result = any(warp, value);
		if (warp.thread_rank() == 0)
			results[warp_id] = warp_result;

		g.sync();
		bool warp_results = true;
		offset_t number_of_warps = g.size() / MAX_NUMBER_OF_THREADS_PER_WARP;
		offset_t local_warp_id = g.thread_rank() % MAX_NUMBER_OF_THREADS_PER_WARP;
		if (local_warp_id < number_of_warps)
			warp_results = results[local_warp_id];

		return any(warp, warp_results);
	}

	template <int tile_sz>
	GPU_DEVICE bool any(block_tile_t<tile_sz> g, bool value)
	{
		return g.any(value);
	}
}
