#include <gstl/grid/numeric/transform_reduce.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/kernel/algorithms/fill.cuh>

struct test_grid_transform_reduce_functor
{
	template <typename T>
	GPU_DEVICE auto operator()(T&& type) -> decltype(type * 2)
	{
		return type * 2;
	}
};

template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp, class UnaryOp>
GPU_GLOBAL void test_grid_transform_reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op, UnaryOp unary_op)
{
	gpu::grid_t grid = gpu::this_grid();
	gpu::transform_reduce(grid, first, last, buffer, *init, binary_op, unary_op);
}

GPU_GLOBAL void test_grid_transform_reduce_result(int* buffer, unsigned int buffer_size, unsigned int number_of_elements)
{
	gpu::block_t block = gpu::this_thread_block();

	int result = (2 * number_of_elements) / buffer_size;
	if (block.thread_rank() < buffer_size)
		ENSURE(buffer[block.thread_rank()] == (block.thread_rank() == 0) ? result + 1 : result);
}

TEST_CASE("GRID TRANSFORM_REDUCE", "[GRID_TRANSFORM_REDUCE][NUMERIC]")
{
	SECTION("GRID transform_reduce")
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
		CHECK(gpu::kernel::launch_kernel(
			test_grid_transform_reduce<int*, int*, int*, decltype(gpu::plus<>{}), decltype(test_grid_transform_reduce_functor{})>,
			cuda::launch_configuration_t{ blocks_per_grid, threads_per_block },
			d_input.get(), d_input.get() + capacity, d_buffer.get(), d_one.get(), gpu::plus<>{}, test_grid_transform_reduce_functor{}
		));
		gpu::kernel::sync();

		gpu::kernel::launch_kernel(
			test_grid_transform_reduce_result,
			cuda::launch_configuration_t{ 1u, 1024u },
			d_buffer.get(), blocks_per_grid, capacity
		);
	}
}
