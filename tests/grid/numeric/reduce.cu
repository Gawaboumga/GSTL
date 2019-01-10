#include <gstl/grid/numeric/reduce.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/kernel/algorithms/fill.cuh>

template <class RandomIt, class RandomOutputIt>
GPU_GLOBAL void test_grid_reduce(RandomIt first, RandomIt last, RandomOutputIt buffer)
{
	gpu::grid_t grid = gpu::this_grid();
	gpu::reduce(grid, first, last, buffer);
}

GPU_GLOBAL void test_grid_reduce_result(int* buffer, unsigned int buffer_size, unsigned int number_of_elements)
{
	gpu::block_t block = gpu::this_thread_block();

	if (block.thread_rank() < buffer_size)
		ENSURE(buffer[block.thread_rank()] == number_of_elements / buffer_size);
}

TEST_CASE("GRID REDUCE", "[GRID_REDUCE][NUMERIC]")
{
	SECTION("GRID reduce")
	{
		auto current_device = cuda::device::current::get();
		unsigned int capacity = 32 * 1024 * 4;
		unsigned int number_of_blocks = 128;
		auto d_input = cuda::memory::device::make_unique<int[]>(current_device, capacity);
		auto d_buffer = cuda::memory::device::make_unique<int[]>(current_device, number_of_blocks);

		auto h_one = std::make_unique<int>(1);
		auto d_one = cuda::memory::device::make_unique<int>(current_device);
		cuda::memory::copy(d_one.get(), h_one.get(), sizeof(int));

		gpu::kernel::fill(d_input.get(), d_input.get() + capacity, d_one.get());
		gpu::kernel::sync();
		unsigned int blocks_per_grid = 32u;
		unsigned int threads_per_block = 256u;
		gpu::kernel::launch_kernel(
			test_grid_reduce<int*, int*>,
			cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
			d_input.get(), d_input.get() + capacity, d_buffer.get()
		);
		gpu::kernel::sync();
		CHECK(gpu::kernel::launch_kernel(
			test_grid_reduce_result,
			cuda::launch_configuration_t{ 1u, threads_per_block },
			d_buffer.get(), blocks_per_grid, capacity
		));
	}
}
