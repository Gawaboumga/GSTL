#include <gstl/kernel/numeric/reduce.cuh>

#include <gstl/grid/numeric/reduce.cuh>
#include <gstl/kernel/launch_kernel.cuh>

#include <gstl/utility/iterator.cuh>

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

		template <class RandomIt, class OutputIt>
		GPU_GLOBAL void block_reduce(RandomIt first, RandomIt last, OutputIt result)
		{
			gpu::block_t block = gpu::this_thread_block();
			unsigned int len = distance(first, last);
			auto group_result = gpu::reduce(block, *(first + block.thread_rank()), len);
			if (block.thread_rank() == 0)
				*result = group_result;
		}

		template <class RandomIt, class OutputIt, class BinaryOp>
		GPU_GLOBAL void block_reduce(RandomIt first, RandomIt last, OutputIt result, BinaryOp binary_op)
		{
			gpu::block_t block = gpu::this_thread_block();
			unsigned int len = distance(first, last);
			auto group_result = gpu::reduce(block, *(first + block.thread_rank()), binary_op, len);
			if (block.thread_rank() == 0)
				*result = group_result;
		}

		template <class RandomIt, class OutputIt>
		void reduce_buffer(RandomIt first, RandomIt last, OutputIt result)
		{
			unsigned int blocks_per_grid = 1u;
			unsigned int threads_per_block = 1024u;
			cuda::launch_configuration_t configuration{ blocks_per_grid, threads_per_block };

			kernel::launch_kernel(
				detail::block_reduce<RandomIt, OutputIt>,
				configuration,
				first, last, result
			);
		}

		template <class RandomIt, class OutputIt, class BinaryOp>
		void reduce_buffer(RandomIt first, RandomIt last, OutputIt result, BinaryOp binary_op)
		{
			unsigned int blocks_per_grid = 1u;
			unsigned int threads_per_block = 1024u;
			cuda::launch_configuration_t configuration{ blocks_per_grid, threads_per_block };

			kernel::launch_kernel(
				detail::block_reduce<RandomIt, OutputIt>,
				configuration,
				first, last, result
			);
		}
	}
}

namespace gpu
{
	namespace kernel
	{
		template <class RandomIt>
		typename std::iterator_traits<RandomIt>::value_type reduce(RandomIt first, RandomIt last)
		{
			unsigned int blocks_per_grid = 64u;
			unsigned int threads_per_block = 1024u;
			cuda::launch_configuration_t configuration{ blocks_per_grid, threads_per_block };
			return reduce(configuration, first, last);
		}

		template <class RandomIt, typename T>
		T reduce(RandomIt first, RandomIt last, T init)
		{
			unsigned int blocks_per_grid = 64u;
			unsigned int threads_per_block = 1024u;
			cuda::launch_configuration_t configuration{ blocks_per_grid, threads_per_block };
			return reduce(configuration, first, last, init);
		}

		template <class RandomIt, typename T, class BinaryOp>
		T reduce(RandomIt first, RandomIt last, T init, BinaryOp binary_op)
		{
			unsigned int blocks_per_grid = 64u;
			unsigned int threads_per_block = 1024u;
			cuda::launch_configuration_t configuration{ blocks_per_grid, threads_per_block };
			return reduce(configuration, first, last, init, binary_op);
		}

		template <class RandomIt>
		typename std::iterator_traits<RandomIt>::value_type reduce(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last)
		{
			using value_type = typename std::iterator_traits<RandomIt>::value_type;
			auto number_of_blocks = configuration.grid_dimensions.volume();
			auto current_device = cuda::device::current::get();
			auto d_temporary_buffer = cuda::memory::device::make_unique<value_type[]>(current_device, number_of_blocks + 1); // last will be the result

			reduce_to_buffer(configuration, first, last, d_temporary_buffer.get());
			current_device.synchronize();

			detail::reduce_buffer(d_temporary_buffer.get(), d_temporary_buffer.get() + number_of_blocks, d_temporary_buffer.get() + number_of_blocks + 1);

			value_type result;
			cuda::memory::copy(d_temporary_buffer.get() + number_of_blocks, &result, sizeof(value_type) * 1);
			return result;
		}

		template <class RandomIt, typename T>
		T reduce(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, T init)
		{
			using value_type = typename std::iterator_traits<RandomIt>::value_type;
			auto number_of_blocks = configuration.grid_dimensions.volume();
			auto current_device = cuda::device::current::get();
			auto d_temporary_buffer = cuda::memory::device::make_unique<value_type[]>(current_device, number_of_blocks + 2); // init + result

			cuda::memory::copy(&init, d_temporary_buffer.get() + number_of_blocks, sizeof(value_type) * 1);

			reduce_to_buffer(configuration, first, last, d_temporary_buffer.get(), init);
			current_device.synchronize();

			detail::reduce_buffer(d_temporary_buffer.get(), d_temporary_buffer.get() + number_of_blocks, d_temporary_buffer.get() + number_of_blocks + 1);

			T result;
			cuda::memory::copy(d_temporary_buffer.get() + number_of_blocks + 1, &result, sizeof(value_type) * 1);
			return result;
		}

		template <class RandomIt, typename T, class BinaryOp>
		T reduce(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, T init, BinaryOp binary_op)
		{
			using value_type = typename std::iterator_traits<RandomIt>::value_type;
			auto number_of_blocks = configuration.grid_dimensions.volume();
			auto current_device = cuda::device::current::get();
			auto d_temporary_buffer = cuda::memory::device::make_unique<value_type[]>(current_device, number_of_blocks + 2); // init + result

			cuda::memory::copy(&init, d_temporary_buffer.get() + number_of_blocks, sizeof(value_type) * 1);

			reduce_to_buffer(configuration, first, last, d_temporary_buffer.get(), init, binary_op);
			current_device.synchronize();

			detail::reduce_buffer(d_temporary_buffer.get(), d_temporary_buffer.get() + number_of_blocks, d_temporary_buffer.get() + number_of_blocks + 1, binary_op);

			T result;
			cuda::memory::copy(d_temporary_buffer.get() + number_of_blocks + 1, &result, sizeof(value_type) * 1);
			return result;
		}

		template <class RandomIt, class RandomOutputIt>
		void reduce_to_buffer(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, RandomOutputIt buffer)
		{
			launch_kernel(
				detail::reduce<RandomIt, RandomOutputIt>,
				configuration,
				first, last, buffer
			);
		}

		template <class RandomIt, class RandomOutputIt, typename T>
		void reduce_to_buffer(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, RandomOutputIt buffer, T init)
		{
			launch_kernel(
				detail::reduce<RandomIt, RandomOutputIt, T>,
				configuration,
				first, last, buffer, init
			);
		}

		template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp>
		void reduce_to_buffer(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op)
		{
			launch_kernel(
				detail::reduce<RandomIt, RandomOutputIt, T, BinaryOp>,
				configuration,
				first, last, buffer, init, binary_op
			);
		}
	}
}
