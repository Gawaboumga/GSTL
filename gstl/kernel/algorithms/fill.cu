#include <gstl/kernel/algorithms/fill.cuh>

#include <gstl/grid/algorithms/fill.cuh>
#include <gstl/kernel/launch_kernel.cuh>

namespace gpu
{
	namespace detail
	{
		template <class RandomIt, typename T>
		GPU_GLOBAL void fill(RandomIt first, RandomIt last, const T* value)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::fill(grid, first, last, *value);
		}
	}
}

namespace gpu
{
	namespace kernel
	{
		template <class RandomIt, typename T>
		void fill(RandomIt first, RandomIt last, const T* value)
		{
			auto current_device = cuda::device::current::get();
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::fill<RandomIt, T>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first, last, value
			);
		}
	}
}
