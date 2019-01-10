#include <gstl/kernel/numeric/reduce.cuh>

#include <gstl/grid/numeric/reduce.cuh>
#include <gstl/kernel/launch_kernel.cuh>

namespace gpu
{
	namespace detail
	{
		template <class RandomIt, class RandomOutputIt>
		GPU_GLOBAL void reduce(RandomIt first, RandomIt last, RandomOutputIt buffer)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::reduce(grid, first, last, buffer);
		}

		template <class RandomIt, class RandomOutputIt, typename T>
		GPU_GLOBAL void reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::reduce(grid, first, last, buffer, init);
		}

		template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp>
		GPU_GLOBAL void reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::reduce(grid, first, last, buffer, init, binary_op);
		}
	}
}

namespace gpu
{
	namespace kernel
	{
		template <class RandomIt, class RandomOutputIt>
		unsigned int reduce(RandomIt first, RandomIt last, RandomOutputIt buffer)
		{
			auto current_device = cuda::device::current::get();
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::reduce<RandomIt, RandomOutputIt>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first, last, buffer
			);

			return blocks_per_grid;
		}

		template <class RandomIt, class RandomOutputIt, typename T>
		unsigned int reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init)
		{
			auto current_device = cuda::device::current::get();
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::reduce<RandomIt, RandomOutputIt, T>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first, last, buffer, init
			);

			return blocks_per_grid;
		}

		template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp>
		unsigned int reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op)
		{
			auto current_device = cuda::device::current::get();
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::reduce<RandomIt, RandomOutputIt, T, BinaryOp>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first, last, buffer, init, binary_op
			);

			return blocks_per_grid;
		}
	}
}
