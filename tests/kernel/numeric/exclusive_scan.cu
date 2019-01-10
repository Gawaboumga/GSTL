#include <gstl/kernel/numeric/exclusive_scan.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/kernel/algorithms/enumerate.cuh>
#include <gstl/kernel/algorithms/fill.cuh>

struct test_kernel_exclusive_scan_functor
{
	GPU_DEVICE void operator()(int value, gpu::offset_t offset)
	{
		ENSURE(value == offset);
	}
};

TEST_CASE("KERNEL EXCLUSIVE SCAN", "[KERNEL_EXCLUSIVE_SCAN][NUMERIC]")
{
	SECTION("KERNEL exclusive_scan")
	{
		auto current_device = cuda::device::current::get();
		unsigned int capacity = 32 * 1024 * 4;
		auto d_input = cuda::memory::device::make_unique<int[]>(current_device, capacity);
		auto d_output = cuda::memory::device::make_unique<int[]>(current_device, capacity);

		auto h_one = std::make_unique<int>(1);
		auto d_one = cuda::memory::device::make_unique<int>(current_device);
		cuda::memory::copy(d_one.get(), h_one.get(), sizeof(int));

		gpu::kernel::fill(d_input.get(), d_input.get() + capacity, d_one.get());
		gpu::kernel::sync();
		gpu::kernel::exclusive_scan(d_input.get(), d_input.get() + capacity, d_output.get());
		gpu::kernel::sync();

		gpu::kernel::enumerate(d_output.get(), d_output.get() + capacity, test_kernel_exclusive_scan_functor{});
	}
}
