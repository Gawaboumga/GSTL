#include <gstl/kernel/numeric/exclusive_scan.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/grid/algorithms/fill.cuh>

template <class RandomIt, typename T, int value>
GPU_GLOBAL void test_kernel_exclusive_scan_prepare_fill(RandomIt first, RandomIt last)
{
	gpu::grid_t grid = gpu::this_grid();
	gpu::fill(grid, first, last, value);
}

TEST_CASE("KERNEL EXCLUSIVE SCAN", "[KERNEL_EXCLUSIVE_SCAN][NUMERIC]")
{
	SECTION("GRID exclusive_scan")
	{
		auto current_device = cuda::device::current::get();
		unsigned int capacity = 1000;
		auto d_input = cuda::memory::device::make_unique<int[]>(current_device, capacity);
		auto d_output = cuda::memory::device::make_unique<int[]>(current_device, capacity);

		CHECK(launch_kernel(test_kernel_exclusive_scan_prepare_fill<int*, int, 1>, d_input.get(), d_input.get() + capacity));
		CHECK(launch_kernel(test_kernel_exclusive_scan_prepare_fill<int*, int, 0>, d_output.get(), d_output.get() + capacity));
		gpu::exclusive_scan(d_input.get(), d_input.get() + 200, d_output.get());
	}
}
