#include <gstl/numeric/exclusive_scan.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/algorithms/is_sorted.cuh>
#include <gstl/containers/array.cuh>
#include <gstl/functional/function_object.cuh>

template <unsigned int number_of_elements>
GPU_GLOBAL void test_exclusive_scan_block()
{
	gpu::block_t block = gpu::this_thread_block();
	GPU_SHARED gpu::array<int, number_of_elements> input;
	GPU_SHARED gpu::array<int, number_of_elements> output;

	gpu::fill(block, input.begin(), input.end(), 1);
	block.sync();

	auto end_ptr = gpu::exclusive_scan(block, input.begin(), input.end(), output.begin(), 0);

	ENSURE(end_ptr == output.end());
	ENSURE(gpu::is_sorted(output.begin(), output.end()));
	ENSURE(*output.begin() == 0);
	ENSURE(*(--end_ptr) == number_of_elements - 1);
}

template <unsigned int number_of_elements>
GPU_GLOBAL void test_exclusive_scan_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	GPU_SHARED gpu::array<int, number_of_elements> input;
	GPU_SHARED gpu::array<int, number_of_elements> output;

	gpu::fill(block, input.begin(), input.end(), 1);
	block.sync();

	if (block.thread_rank() < warp.size())
	{
		auto end_ptr = gpu::exclusive_scan(warp, input.begin(), input.end(), output.begin(), 0);

		ENSURE(end_ptr == output.end());
		ENSURE(gpu::is_sorted(output.begin(), output.end()));
		ENSURE(*output.begin() == 0);
		ENSURE(*(--end_ptr) == number_of_elements - 1);
	}
}

GPU_GLOBAL void test_exclusive_scan_one_value_block()
{
	gpu::block_t block = gpu::this_thread_block();
	int init = 3;
	gpu::offset_t value = gpu::exclusive_scan(block, 1, init);
	ENSURE(value == block.thread_rank() + init);

	GPU_SHARED gpu::array<int, gpu::MAX_NUMBER_OF_THREADS_PER_BLOCK> data;
	data[block.thread_rank()] = block.thread_rank() % 4 + 1;

	value = gpu::exclusive_scan(block, data[block.thread_rank()], 0);
	gpu::array<int, 4> tmp = { 0, 1, 3, 6 };
	ENSURE(value == 10 * (block.thread_rank() / 4) + tmp[block.thread_rank() % 4]);
}

GPU_GLOBAL void test_exclusive_scan_one_value_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	if (block.thread_rank() >= warp.size())
		return;

	int init = 3;
	gpu::offset_t value = gpu::exclusive_scan(warp, 1, init);
	ENSURE(value == warp.thread_rank() + init);

	GPU_SHARED gpu::array<int, 32> data;
	GPU_SHARED gpu::array<int, 32> output;
	if (warp.thread_rank() == 0)
	{
		data = { 33, 33, 33, 33, 34, 34, 0, -1, 1, 1, 30, 34, 0, 25, 7, 20, -3, -8, 14, -23, -39,-47, -54, -49, 19, -5, 12, -43, 18, 24, 17, -56 };
		output = { 0, 33, 66, 99, 132, 166, 200, 200, 199, 200, 201, 231, 265, 265, 290, 297, 317, 314, 306, 320, 297, 258, 211, 157, 108, 127, 122, 134, 91, 109, 133, 150 };
	}
	warp.sync();

	int result = gpu::exclusive_scan(warp, data[warp.thread_rank()], 0);
	ENSURE(result == output[warp.thread_rank()]);
}

TEST_CASE("EXCLUSIVE SCAN", "[EXCLUSIVE_SCAN][NUMERIC]")
{
	SECTION("BLOCK exclusive_scan")
	{
		CHECK(launch(test_exclusive_scan_block<200>));
		CHECK(launch(test_exclusive_scan_block<256>));
		CHECK(launch(test_exclusive_scan_block<2000>));
		CHECK(launch(test_exclusive_scan_block<2048>));
	}

	SECTION("WARP exclusive_scan")
	{
		CHECK(launch(test_exclusive_scan_warp<200>));
		CHECK(launch(test_exclusive_scan_warp<256>));
		CHECK(launch(test_exclusive_scan_warp<2000>));
		CHECK(launch(test_exclusive_scan_warp<2048>));
	}

	SECTION("BLOCK exclusive_scan_one_value")
	{
		CHECK(launch(test_exclusive_scan_one_value_block));
	}

	SECTION("WARP exclusive_scan_one_value")
	{
		CHECK(launch(test_exclusive_scan_one_value_warp));
	}
}
