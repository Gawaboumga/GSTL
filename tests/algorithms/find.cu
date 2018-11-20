#include <gstl/algorithms/find.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_find_block()
{
	gpu::block_t block = gpu::this_thread_block();

	int winner = (block_size * 3) / 4;
	GPU_SHARED gpu::array<int, block_size> in;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	block.sync();

	auto found_it = gpu::find(block, in.begin(), in.end(), value);
	ENSURE(found_it == in.begin());

	found_it = gpu::find(block, in.begin(), in.end(), value + 1);
	ENSURE(found_it == in.end());

	if (block.thread_rank() == 0)
		in[winner] = value + 1;

	found_it = gpu::find(block, in.begin(), in.end(), value + 1);
	ENSURE(found_it == in.begin() + winner);
}

template <unsigned int block_size>
GPU_GLOBAL void test_find_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	int winner = (block_size * 3) / 4;
	GPU_SHARED gpu::array<int, block_size> in;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	block.sync();

	auto found_it = gpu::find(block, in.begin(), in.end(), value);
	ENSURE(found_it == in.begin());

	found_it = gpu::find(block, in.begin(), in.end(), value + 1);
	ENSURE(found_it == in.end());

	if (block.thread_rank() == 0)
		in[winner] = value + 1;

	found_it = gpu::find(warp, in.begin(), in.end(), value + 1);
	ENSURE(found_it == in.begin() + winner);
}

TEST_CASE("FIND", "[FIND][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_find_block<200>));
		CHECK(launch(test_find_block<256>));
		CHECK(launch(test_find_block<2000>));
		CHECK(launch(test_find_block<2048>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_find_warp<200>));
		CHECK(launch(test_find_warp<256>));
		CHECK(launch(test_find_warp<2000>));
		CHECK(launch(test_find_warp<2048>));
	}
}
