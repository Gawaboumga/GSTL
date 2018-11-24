#include <gstl/algorithms/adjacent_find.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_adjacent_find_block()
{
	gpu::block_t block = gpu::this_thread_block();
	GPU_SHARED gpu::array<int, block_size> data;

	gpu::fill(block, data.begin(), data.end(), 1);
	block.sync();

	auto it = gpu::adjacent_find(block, data.begin(), data.end(), gpu::equal_to<>());
	ENSURE(it == data.end());

	if (block.thread_rank() == 0)
		data[(block_size * 3) / 4] = 2;

	it = gpu::adjacent_find(block, data.begin(), data.end(), gpu::equal_to<>());
	ENSURE(it == (&data[(block_size * 3) / 4] - 1));
}

template <unsigned int block_size>
GPU_GLOBAL void test_adjacent_find_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	GPU_SHARED gpu::array<int, block_size> data;

	gpu::fill(block, data.begin(), data.end(), 1);
	block.sync();

	if (block.thread_rank() >= warp.size())
		return;

	auto it = gpu::adjacent_find(warp, data.begin(), data.end(), gpu::equal_to<>());
	ENSURE(it == data.end());

	if (block.thread_rank() == 0)
		data[(block_size * 3) / 4] = 2;

	it = gpu::adjacent_find(warp, data.begin(), data.end(), gpu::equal_to<>());
	ENSURE(it == (&data[(block_size * 3) / 4] - 1));
}

TEST_CASE("ADJACENT_FIND", "[ADJACENT_FIND][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_adjacent_find_block<200>));
		CHECK(launch(test_adjacent_find_block<256>));
		CHECK(launch(test_adjacent_find_block<2000>));
		CHECK(launch(test_adjacent_find_block<2048>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_adjacent_find_warp<200>));
		CHECK(launch(test_adjacent_find_warp<256>));
		CHECK(launch(test_adjacent_find_warp<2000>));
		CHECK(launch(test_adjacent_find_warp<2048>));
	}
}
