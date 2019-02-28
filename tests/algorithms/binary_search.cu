#include <gstl/algorithms/binary_search.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/equal.cuh>
#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_binary_search_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> in;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	block.sync();
	gpu::fill(block, &in[(block_size * 3) / 4], in.end(), value + 1);
	block.sync();

	auto it = gpu::lower_bound(block, in.begin(), in.end(), value + 1);
	block.sync();

	ENSURE(it == &in[(block_size * 3) / 4]);
}

template <unsigned int block_size>
GPU_GLOBAL void test_binary_search_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> in;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	block.sync();
	gpu::fill(block, &in[(block_size * 3) / 4], in.end(), value + 1);
	block.sync();

	auto it = gpu::lower_bound(warp, in.begin(), in.end(), value + 1);
	block.sync();

	ENSURE(it == &in[(block_size * 3) / 4]);
}

GPU_GLOBAL void test_upper_bound_block()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	constexpr unsigned int block_size = 20;
	GPU_SHARED gpu::array<int, block_size> in;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	block.sync();
	gpu::fill(block, &in[(block_size * 2) / 4], &in[(block_size * 3) / 4], value + 1);
	block.sync();

	auto it = gpu::upper_bound(warp, in.begin(), in.end(), value);
	block.sync();

	ENSURE(it == &in[(block_size * 2) / 4]);

	it = gpu::upper_bound(block, in.begin(), in.end(), value);
	block.sync();

	ENSURE(it == &in[(block_size * 2) / 4]);
}

TEST_CASE("BINARY_SEARCH", "[BINARY_SEARCH][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_binary_search_block<200>));
		CHECK(launch(test_binary_search_block<256>));
		CHECK(launch(test_binary_search_block<4000>));
		CHECK(launch(test_binary_search_block<4096>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_binary_search_warp<200>));
		CHECK(launch(test_binary_search_warp<256>));
		CHECK(launch(test_binary_search_warp<4000>));
		CHECK(launch(test_binary_search_warp<4096>));
	}

	SECTION("Upper bound")
	{
		CHECK(launch(test_upper_bound_block));
	}
}
