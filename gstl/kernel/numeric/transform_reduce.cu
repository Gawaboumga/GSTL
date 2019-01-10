#include <gstl/kernel/numeric/transform_reduce.cuh>

#include <gstl/grid/numeric/transform_reduce.cuh>
#include <gstl/kernel/launch_kernel.cuh>

namespace gpu
{
	namespace detail
	{
		template <class RandomIt1, class RandomIt2, class RandomOutputIt, typename T>
		GPU_GLOBAL void transform_reduce(RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomOutputIt buffer, T init)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::transform_reduce(grid, first1, last1, first2, buffer, *init);
		}

		template <class RandomIt1, class RandomIt2, class RandomOutputIt, typename T, class BinaryOp1, class BinaryOp2>
		GPU_GLOBAL void transform_reduce(RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomOutputIt buffer, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::transform_reduce(grid, first1, last1, first2, buffer, *init, binary_op1, binary_op2);
		}

		template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp, class UnaryOp>
		GPU_GLOBAL void transform_reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op, UnaryOp unary_op)
		{
			gpu::grid_t grid = gpu::this_grid();
			gpu::transform_reduce(grid, first, last, buffer, *init, binary_op, unary_op);
		}
	}
}

namespace gpu
{
	namespace kernel
	{
		template <class RandomIt1, class RandomIt2, class RandomOutputIt, typename T>
		unsigned int transform_reduce(RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomOutputIt buffer, T init)
		{
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::transform_reduce<RandomIt1, RandomIt2, RandomOutputIt, T>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first1, last1, first2, buffer, init
			);

			return blocks_per_grid;
		}

		template <class RandomIt1, class RandomIt2, class RandomOutputIt, typename T, class BinaryOp1, class BinaryOp2>
		unsigned int transform_reduce(RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomOutputIt buffer, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2)
		{
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::transform_reduce<RandomIt1, RandomIt2, RandomOutputIt, T, BinaryOp1, BinaryOp2>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first1, last1, first2, buffer, init, binary_op1, binary_op2
			);

			return blocks_per_grid;
		}

		template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp, class UnaryOp>
		unsigned int transform_reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op, UnaryOp unary_op)
		{
			unsigned int blocks_per_grid = 32u;
			unsigned int threads_per_block = 1024u;

			launch_kernel(
				detail::transform_reduce<RandomIt, RandomOutputIt, T, BinaryOp, UnaryOp>,
				cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
				first, last, buffer, init, binary_op, unary_op
			);

			return blocks_per_grid;
		}
	}
}
