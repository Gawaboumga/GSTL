#include <gstl/utility/ballot.cuh>

#include <gstl/containers/array.cuh>
#include <gstl/math/intrinsics.cuh>
#include <gstl/utility/limits.cuh>

namespace gpu
{
	namespace detail
	{
		template <typename T, bool pointer_like = std::is_pointer<T>::value>
		struct shuffle
		{
		};

		template <>
		struct shuffle<std::uintptr_t, false>
		{
			template <unsigned int tile_sz>
			GPU_DEVICE std::uintptr_t operator()(gpu::block_tile_t<tile_sz> g, std::uintptr_t var, unsigned int thid) const noexcept
			{
				return g.shfl(var, thid);
			}
		};

		template <typename T>
		struct shuffle<T, true>
		{
			template <unsigned int tile_sz>
			GPU_DEVICE T operator()(gpu::block_tile_t<tile_sz> g, T var, unsigned int thid) const noexcept
			{
				return reinterpret_cast<T>(shuffle<std::uintptr_t>{}(g, reinterpret_cast<std::uintptr_t>(var), thid));
			}
		};

		template <typename T>
		struct shuffle<T, false>
		{
			template <unsigned int tile_sz>
			GPU_DEVICE T operator()(gpu::block_tile_t<tile_sz> g, T var, unsigned int thid) const noexcept
			{
				return g.shfl(var, thid);
			}
		};

		template <class BlockTile, typename T>
		GPU_DEVICE inline T sum(BlockTile g, T value)
		{
			for (int mask = g.size() / 2; mask > 0; mask >>= 1)
				value += g.shfl_xor(value, mask);
			return value;
		}
	}

	GPU_DEVICE inline bool all(block_t g, bool value)
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
		g.sync();
		
		return all(warp, warp_results);
	}

	template <class BlockTile>
	GPU_DEVICE inline bool all(BlockTile g, bool value)
	{
		return g.all(value);
	}

	GPU_DEVICE inline bool any(block_t g, bool value)
	{
		GPU_SHARED gpu::array<bool, MAX_NUMBER_OF_WARPS_PER_BLOCK> results;

		offset_t warp_id = g.thread_rank() / MAX_NUMBER_OF_THREADS_PER_WARP;
		block_tile_t<MAX_NUMBER_OF_THREADS_PER_WARP> warp = tiled_partition<MAX_NUMBER_OF_THREADS_PER_WARP>(g);
		bool warp_result = any(warp, value);
		if (warp.thread_rank() == 0)
			results[warp_id] = warp_result;
		g.sync();

		bool warp_results = false;
		offset_t number_of_warps = g.size() / MAX_NUMBER_OF_THREADS_PER_WARP;
		offset_t local_warp_id = g.thread_rank() % MAX_NUMBER_OF_THREADS_PER_WARP;
		if (local_warp_id < number_of_warps)
			warp_results = results[local_warp_id];
		g.sync();

		return any(warp, warp_results);
	}

	template <class BlockTile>
	GPU_DEVICE inline bool any(BlockTile g, bool value)
	{
		return g.any(value);
	}

	GPU_DEVICE inline unsigned int count(block_t g, bool value)
	{
		GPU_SHARED gpu::array<unsigned int, MAX_NUMBER_OF_WARPS_PER_BLOCK> results;

		offset_t warp_id = g.thread_rank() / MAX_NUMBER_OF_THREADS_PER_WARP;
		block_tile_t<MAX_NUMBER_OF_THREADS_PER_WARP> warp = tiled_partition<MAX_NUMBER_OF_THREADS_PER_WARP>(g);
		auto warp_result = count(warp, value);
		if (warp.thread_rank() == 0)
			results[warp_id] = warp_result;
		g.sync();

		auto warp_results = 0;
		offset_t number_of_warps = g.size() / MAX_NUMBER_OF_THREADS_PER_WARP;
		offset_t local_warp_id = g.thread_rank() % MAX_NUMBER_OF_THREADS_PER_WARP;
		if (local_warp_id < number_of_warps)
			warp_results = results[local_warp_id];
		g.sync();

		return detail::sum(warp, warp_results);
	}

	template <class BlockTile>
	GPU_DEVICE inline unsigned int count(BlockTile g, bool value)
	{
		return popc(g.ballot(value));
	}

	GPU_DEVICE inline offset_t first_index(block_t g, bool value, offset_t from)
	{
	#ifdef GPU_DEBUG
		ENSURE(from < g.size());
	#endif // GPU_DEBUG

		GPU_SHARED gpu::array<bool, MAX_NUMBER_OF_WARPS_PER_BLOCK> results;
		GPU_SHARED gpu::array<offset_t, MAX_NUMBER_OF_WARPS_PER_BLOCK> indices;

		offset_t warp_id = g.thread_rank() / MAX_NUMBER_OF_THREADS_PER_WARP;
		block_tile_t<MAX_NUMBER_OF_THREADS_PER_WARP> warp = tiled_partition<MAX_NUMBER_OF_THREADS_PER_WARP>(g);
		offset_t warp_result = first_index(warp, value, from % warp.size());
		if (warp.thread_rank() == 0)
		{
			results[warp_id] = warp_result != warp.size() && warp_id >= from / warp.size();
			indices[warp_id] = warp_id * warp.size() + warp_result;
		}
		g.sync();

		bool warp_results = false;
		offset_t number_of_warps = g.size() / warp.size();
		offset_t local_warp_id = g.thread_rank() % warp.size();
		if (local_warp_id < number_of_warps)
			warp_results = results[local_warp_id];
		g.sync();

		offset_t index = first_index(warp, warp_results);
		if (index == warp.size())
			return g.size();
		else
			return indices[index];
	}

	template <class BlockTile>
	GPU_DEVICE inline offset_t first_index(BlockTile g, bool value)
	{
		unsigned int mask = g.ballot(value);
		if (!mask)
			return g.size();
		else
			return ffs(mask) - 1;
	}

	template <class BlockTile>
	GPU_DEVICE inline offset_t first_index(BlockTile g, bool value, offset_t from)
	{
	#ifdef GPU_DEBUG
		ENSURE(from < g.size());
	#endif // GPU_DEBUG

		unsigned int mask = g.ballot(value);
		if (from > 0)
			mask &= ~((1u << from) - 1u);

		if (!mask)
			return g.size();
		else
			return ffs(mask) - 1;
	}

	template <typename T>
	GPU_DEVICE inline T shfl(block_t g, T value, unsigned int thid)
	{
	#ifdef GPU_DEBUG
		ENSURE(thid < g.size());
	#endif // GPU_DEBUG

		GPU_SHARED T shared_value;
		if (g.thread_rank() == thid)
			shared_value = value;
		g.sync();
		return shared_value;
	}

	template <class BlockTile, typename T>
	GPU_DEVICE inline T shfl(BlockTile g, T value, unsigned int thid)
	{
	#ifdef GPU_DEBUG
		ENSURE(thid < g.size());
	#endif // GPU_DEBUG

		return detail::shuffle<T>{}(g, value, thid);
	}
}
