#include <gstl/numeric/inclusive_scan.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/algorithms/sort.cuh>
#include <gstl/containers/array.cuh>
#include <gstl/functional/function_object.cuh>

template <unsigned int number_of_elements>
GPU_GLOBAL void test_inclusive_scan_block()
{
	gpu::block_t block = gpu::this_thread_block();
	GPU_SHARED gpu::array<int, number_of_elements> input;
	GPU_SHARED gpu::array<int, number_of_elements> output;

	gpu::fill(block, input.begin(), input.end(), 1);
	block.sync();

	auto end_ptr = gpu::inclusive_scan(block, input.begin(), input.end(), output.begin(), 0);

	ENSURE(end_ptr == output.end());
	ENSURE(gpu::is_sorted(output.begin(), output.end()));
	ENSURE(*output.begin() == 1);
	ENSURE(*(--end_ptr) == number_of_elements);
}

template <unsigned int number_of_elements>
GPU_GLOBAL void test_inclusive_scan_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	GPU_SHARED gpu::array<int, number_of_elements> input;
	GPU_SHARED gpu::array<int, number_of_elements> output;

	gpu::fill(block, input.begin(), input.end(), 1);
	block.sync();

	if (block.thread_rank() < warp.size())
	{
		auto end_ptr = gpu::inclusive_scan(warp, input.begin(), input.end(), output.begin(), 0);

		ENSURE(end_ptr == output.end());
		ENSURE(gpu::is_sorted(output.begin(), output.end()));
		ENSURE(*output.begin() == 1);
		ENSURE(*(--end_ptr) == number_of_elements);
	}
}

GPU_GLOBAL void test_inclusive_scan_one_value_block()
{
	gpu::block_t block = gpu::this_thread_block();
	int init = 3;
	gpu::offset_t value = gpu::inclusive_scan(block, 1, init);
	ENSURE(value == block.thread_rank() + 1 + init);
}

GPU_GLOBAL void test_inclusive_scan_one_value_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	if (block.thread_rank() >= warp.size())
		return;

	int init = 3;
	gpu::offset_t value = gpu::inclusive_scan(warp, 1, init);
	ENSURE(value == warp.thread_rank() + 1 + init);
}

TEST_CASE("INCLUSIVE SCAN", "[INCLUSIVE_SCAN][NUMERIC]")
{
	SECTION("BLOCK inclusive_scan")
	{
		CHECK(launch(test_inclusive_scan_block<200>));
		CHECK(launch(test_inclusive_scan_block<256>));
		CHECK(launch(test_inclusive_scan_block<2000>));
		CHECK(launch(test_inclusive_scan_block<2048>));
	}

	SECTION("WARP inclusive_scan")
	{
		CHECK(launch(test_inclusive_scan_warp<200>));
		CHECK(launch(test_inclusive_scan_warp<256>));
		CHECK(launch(test_inclusive_scan_warp<2000>));
		CHECK(launch(test_inclusive_scan_warp<2048>));
	}

	SECTION("BLOCK inclusive_scan_one_value")
	{
		CHECK(launch(test_inclusive_scan_one_value_block));
	}

	SECTION("WARP inclusive_scan_one_value")
	{
		CHECK(launch(test_inclusive_scan_one_value_warp));
	}
}
