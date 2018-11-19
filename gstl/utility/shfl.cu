#include <gstl/utility/shfl.cuh>

namespace gpu
{
	template <typename T>
	GPU_DEVICE T shfl(block_t g, T value, unsigned int thid)
	{
		GPU_SHARED T shared_value;
		if (g.thread_rank() == thid)
			shared_value = value;
		g.sync();
		return shared_value;
	}

	template <class BlockTile, typename T>
	GPU_DEVICE T shfl(BlockTile g, T value, unsigned int thid)
	{
	#ifdef GPU_DEBUG
		ENSURE(thid < MAX_NUMBER_OF_THREADS_PER_WARP);
	#endif // GPU_DEBUG
		return g.shfl(value, thid);
	}
}
