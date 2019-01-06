#include <gstl/kernel/numeric/exclusive_scan.cuh>

#include <gstl/grid/load_balance.cuh>
#include <gstl/numeric/exclusive_scan.cuh>
#include <gstl/numeric/reduce.cuh>

#include <cuda/api_wrappers.hpp>
#include <iostream>
#include <numeric>

namespace gpu
{
	namespace detail
	{
		template <typename T>
		GPU_GLOBAL void exclusive_scan(T* results_buffer, unsigned int buffer_size)
		{
			gpu::block_t block = gpu::this_thread_block();
			gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

			if (block.thread_rank() >= warp.size())
				return;

			T result;
			if (block.thread_rank() < buffer_size)
				result = results_buffer[block.thread_rank()];
			else
				result = 0;

			result = gpu::exclusive_scan(warp, result, 0);
			results_buffer[block.thread_rank()] = result;
		}

		template <class RandomIt, class OutputIt, typename T>
		GPU_GLOBAL void exclusive_scan(RandomIt first, RandomIt last, OutputIt d_first, T* buffer)
		{
			gpu::grid_t grid = gpu::this_grid();
			auto pair_iterators = gpu::load_balance(grid, first, last);

			gpu::block_t block = gpu::this_thread_block();
			gpu::exclusive_scan(block, pair_iterators.first, pair_iterators.second, d_first + distance(first, pair_iterators.first), buffer[block.group_index().x]);
		}

		template <class RandomIt, typename T>
		GPU_GLOBAL void reduce(RandomIt first, RandomIt last, T* buffer)
		{
			gpu::grid_t grid = gpu::this_grid();
			auto pair_iterators = gpu::load_balance(grid, first, last);

			gpu::block_t block = gpu::this_thread_block();
			auto result = gpu::reduce(block, pair_iterators.first, pair_iterators.second);
			if (block.thread_rank() == 0)
				buffer[block.group_index().x] = result;
		}
	}
}

namespace gpu
{
	namespace detail
	{
		template <class Function, class... Args>
		inline bool launch_kernel(Function f, cuda::launch_configuration_t launch_configuration, Args&&... args)
		{
			cuda::launch(
				f,
				launch_configuration,
				std::forward<Args>(args)...
			);

			auto status = cuda::outstanding_error::get();
			cuda::throw_if_error(status, "Failed to launch kernel");

			return true;
		}
	}

	template <class RandomIt, class OutputIt>
	OutputIt exclusive_scan(RandomIt first, RandomIt last, OutputIt d_first)
	{
		auto current_device = cuda::device::current::get();
		unsigned int blocks_per_grid = 6u;
		unsigned int threads_per_block = 64u;
		using result_type = typename std::remove_reference<decltype(*first)>::type;
		auto d_results = cuda::memory::device::make_unique<result_type[]>(current_device, 32u);

		detail::launch_kernel(
			detail::reduce<RandomIt, result_type>,
			cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
			first, last, d_results.get()
		);
		cuda::device::current::get().synchronize();

		detail::launch_kernel(
			detail::exclusive_scan<result_type>,
			cuda::launch_configuration_t{ 1u, threads_per_block },
			d_results.get(), blocks_per_grid
		);
		cuda::device::current::get().synchronize();

		detail::launch_kernel(
			detail::exclusive_scan<RandomIt, OutputIt, result_type>,
			cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
			first, last, d_first, d_results.get()
		);
		cuda::device::current::get().synchronize();

		auto h_res = std::make_unique<result_type[]>(200);
		cuda::memory::copy(h_res.get(), d_first, sizeof(result_type) * 200);
		for (int i = 0; i != 200; ++i)
			std::cout << h_res[i] << " ";

		return d_first + (last - first);
	}
}
