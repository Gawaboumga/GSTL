#include <gstl/kernel/numeric/reduce.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/kernel/algorithms/fill.cuh>

GPU_GLOBAL void test_kernel_reduce(int* buffer, unsigned int buffer_size, unsigned int number_of_elements)
{
	gpu::block_t block = gpu::this_thread_block();

	if (block.thread_rank() < buffer_size)
		ENSURE(buffer[block.thread_rank()] == number_of_elements / buffer_size);
}

TEST_CASE("KERNEL REDUCE", "[KERNEL_REDUCE][NUMERIC]")
{
	SECTION("KERNEL reduce")
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
		cuda::launch_configuration_t configuration{ number_of_blocks, 1024 };
		gpu::kernel::reduce_to_buffer(configuration, d_input.get(), d_input.get() + capacity, d_buffer.get());

		gpu::kernel::launch_kernel(
			test_kernel_reduce,
			cuda::launch_configuration_t{ 1u, 1024u },
			d_buffer.get(), number_of_blocks, capacity
		);
	}
}
