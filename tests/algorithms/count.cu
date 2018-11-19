#include <gstl/algorithms/count.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/algorithms/generate.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_DEVICE void test_count_block(bool only_ones)
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::offset_t thid = block.thread_rank();
	GPU_SHARED gpu::array<int, block_size> data;

	if (!only_ones)
		gpu::generate(block, data.begin(), data.end(), [thid]() { return thid % 2; });
	else
		gpu::fill(block, data.begin(), data.end(), 1);
	block.sync();

	gpu::group_result<int> g_result = gpu::count(block, data.begin(), data.end(), 1);
	int result = g_result.broadcast(block);
	if (!only_ones)
	{
		ENSURE(result == (block_size / 2));
	}
	else
	{
		ENSURE(result == block_size);
	}
}

template <unsigned int block_size>
GPU_GLOBAL void test_count_block_full()
{
	test_count_block<block_size>(true);
}

template <unsigned int block_size>
GPU_GLOBAL void test_count_block_half()
{
	test_count_block<block_size>(false);
}

template <unsigned int block_size>
GPU_DEVICE void test_count_warp(bool only_ones)
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	gpu::offset_t thid = block.thread_rank();
	GPU_SHARED gpu::array<int, block_size> data;

	if (!only_ones)
		gpu::generate(block, data.begin(), data.end(), [thid]() { return thid % 2; });
	else
		gpu::fill(block, data.begin(), data.end(), 1);
	block.sync();

	gpu::group_result<int> g_result = gpu::count(warp, data.begin(), data.end(), 1);
	int result = g_result.broadcast(block);
	if (!only_ones)
	{
		ENSURE(result == (block_size / 2));
	}
	else
	{
		ENSURE(result == block_size);
	}
}

template <unsigned int block_size>
GPU_GLOBAL void test_count_warp_full()
{
	test_count_warp<block_size>(true);
}

template <unsigned int block_size>
GPU_GLOBAL void test_count_warp_half()
{
	test_count_warp<block_size>(false);
}

TEST_CASE("COUNT", "[COUNT][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_count_block_full<200>));
		CHECK(launch(test_count_block_full<256>));
		CHECK(launch(test_count_block_full<2000>));
		CHECK(launch(test_count_block_full<2048>));

		CHECK(launch(test_count_block_half<200>));
		CHECK(launch(test_count_block_half<256>));
		CHECK(launch(test_count_block_half<2000>));
		CHECK(launch(test_count_block_half<2048>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_count_warp_full<200>));
		CHECK(launch(test_count_warp_full<256>));
		CHECK(launch(test_count_warp_full<2000>));
		CHECK(launch(test_count_warp_full<2048>));

		CHECK(launch(test_count_warp_half<200>));
		CHECK(launch(test_count_warp_half<256>));
		CHECK(launch(test_count_warp_half<2000>));
		CHECK(launch(test_count_warp_half<2048>));
	}
}
