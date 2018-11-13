#include <gstl/algorithms/for_each.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/equal.cuh>
#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_for_each_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> in;
	GPU_SHARED gpu::array<int, block_size> out;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	gpu::fill(block, out.begin(), out.end(), value + 1);

	block.sync();

	gpu::for_each(block, in.begin(), in.end(), [](int& v) {
		v += 1;
	});

	block.sync();

	ENSURE(gpu::equal(block, in.begin(), in.end(), out.begin()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_for_each_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> in;
	GPU_SHARED gpu::array<int, block_size> out;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	gpu::fill(block, out.begin(), out.end(), value + 1);

	block.sync();

	if (block.thread_rank() < warp.size())
	{
		gpu::for_each(warp, in.begin(), in.end(), [](int& v) {
			v += 1;
		});
	}

	block.sync();

	ENSURE(gpu::equal(block, in.begin(), in.end(), out.begin()));
}

TEST_CASE("FOR_EACH", "[FOR_EACH][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_for_each_block<200>));
		CHECK(launch(test_for_each_block<256>));
		CHECK(launch(test_for_each_block<4000>));
		CHECK(launch(test_for_each_block<4096>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_for_each_warp<200>));
		CHECK(launch(test_for_each_warp<256>));
		CHECK(launch(test_for_each_warp<4000>));
		CHECK(launch(test_for_each_warp<4096>));
	}
}
