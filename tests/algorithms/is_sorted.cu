#include <gstl/algorithms/is_sorted.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/enumerate.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_is_sorted_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> data;

	gpu::enumerate(block, data.begin(), data.end(), [](int& value, gpu::offset_t id) {
		value = id;
	});
	block.sync();

	ENSURE(gpu::is_sorted(block, data.begin(), data.end()));
	block.sync();

	if (block.thread_rank() == 0)
		data[(block_size * 3) / 4] = 0;
	block.sync();

	ENSURE(!gpu::is_sorted(block, data.begin(), data.end()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_is_sorted_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> data;

	gpu::enumerate(block, data.begin(), data.end(), [](int& value, gpu::offset_t id) {
		value = id;
	});
	block.sync();

	ENSURE(gpu::is_sorted(warp, data.begin(), data.end()));
	block.sync();

	if (block.thread_rank() == 0)
		data[(block_size * 3) / 4] = 0;
	block.sync();

	ENSURE(!gpu::is_sorted(warp, data.begin(), data.end()));
}

TEST_CASE("IS_SORTED", "[IS_SORTED][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_is_sorted_block<200>));
		CHECK(launch(test_is_sorted_block<256>));
		CHECK(launch(test_is_sorted_block<2000>));
		CHECK(launch(test_is_sorted_block<2048>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_is_sorted_warp<200>));
		CHECK(launch(test_is_sorted_warp<256>));
		CHECK(launch(test_is_sorted_warp<2000>));
		CHECK(launch(test_is_sorted_warp<2048>));
	}
}
