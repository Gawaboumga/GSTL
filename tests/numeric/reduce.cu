#include <gstl/numeric/reduce.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int number_of_elements>
GPU_GLOBAL void test_reduce_block()
{
	gpu::block_t block = gpu::this_thread_block();
	GPU_SHARED gpu::array<int, number_of_elements> data;

	gpu::fill(block, data.begin(), data.end(), 1);
	block.sync();

	gpu::group_result<int> g_result = gpu::reduce(block, data.begin(), data.end());
	int result = g_result.broadcast(block);
	ENSURE(result == number_of_elements);
}

template <unsigned int number_of_elements>
GPU_GLOBAL void test_reduce_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	GPU_SHARED gpu::array<int, number_of_elements> data;

	gpu::fill(block, data.begin(), data.end(), 1);
	block.sync();

	if (block.thread_rank() < warp.size())
	{
		gpu::group_result<int> g_result = gpu::reduce(warp, data.begin(), data.end());
		int result = g_result.broadcast(warp);
		ENSURE(result == number_of_elements);
	}
}

TEST_CASE("REDUCE", "[REDUCE][NUMERIC]")
{
	SECTION("BLOCK reduce")
	{
		CHECK(launch(test_reduce_block<200>));
		CHECK(launch(test_reduce_block<256>));
		CHECK(launch(test_reduce_block<2000>));
		CHECK(launch(test_reduce_block<2048>));
	}

	SECTION("WARP reduce")
	{
		CHECK(launch(test_reduce_warp<200>));
		CHECK(launch(test_reduce_warp<256>));
		CHECK(launch(test_reduce_warp<2000>));
		CHECK(launch(test_reduce_warp<2048>));
	}
}
