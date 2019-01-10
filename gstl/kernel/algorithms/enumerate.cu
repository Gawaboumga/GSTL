#include <gstl/kernel/algorithms/enumerate.cuh>

#include <gstl/grid/algorithms/enumerate.cuh>
#include <gstl/kernel/launch_kernel.cuh>

namespace gpu
{
	namespace detail
	{
		template <class RandomIt, class Function>
		GPU_GLOBAL void enumerate(RandomIt first, RandomIt last, Function f)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::enumerate(grid, first, last, f);
		}

		template <class RandomIt, class Size, class Function>
		GPU_GLOBAL void enumerate(RandomIt first, Size n, Function f)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::enumerate(grid, first, n, f);
		}
	}
}

namespace gpu
{
	namespace kernel
	{
		template <class RandomIt, class Function>
		void enumerate(RandomIt first, RandomIt last, Function f)
		{
			auto current_device = cuda::device::current::get();
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::enumerate<RandomIt, Function>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first, last, f
			);
		}

		template <class RandomIt, class Size, class Function>
		void enumerate(RandomIt first, Size n, Function f)
		{
			auto current_device = cuda::device::current::get();
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::enumerate<RandomIt, Size, Function>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first, n, f
			);
		}
	}
}
