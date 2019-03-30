#include <gstl/algorithms/inplace_merge.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/equal.cuh>
#include <gstl/algorithms/enumerate.cuh>
#include <gstl/algorithms/fill.cuh>
#include <gstl/algorithms/is_sorted.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_inplace_merge_block_full_array()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, 2 * block_size> data;
	auto middle = data.begin() + block_size;

	int value = 5;
	gpu::fill(block, data.begin(), middle, value);
	gpu::fill(block, middle, data.end(), value + 1);
	block.sync();

	gpu::inplace_merge(block, data.begin(), middle, data.end());
	block.sync();

	ENSURE(gpu::is_sorted(block, data.begin(), data.end()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_inplace_merge_warp_full_array()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, 2 * block_size> data;
	auto middle = data.begin() + block_size;

	int value = 5;
	gpu::fill(block, data.begin(), middle, value);
	gpu::fill(block, middle, data.end(), value + 1);
	block.sync();

	if (block.thread_rank() < warp.size())
		gpu::inplace_merge(warp, data.begin(), middle, data.end());
	block.sync();

	ENSURE(gpu::is_sorted(block, data.begin(), data.end()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_inplace_merge_block_monotonous()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, 2 * block_size> data;
	auto middle = data.begin() + block_size;

	gpu::enumerate(block, data.begin(), middle, [](int& value, gpu::offset_t id) {
		value = 2 * id;
	});
	gpu::enumerate(block, middle, data.end(), [](int& value, gpu::offset_t id) {
		value = id;
	});
	block.sync();

	gpu::inplace_merge(block, data.begin(), middle, data.end());
	block.sync();

	ENSURE(gpu::is_sorted(block, data.begin(), data.end()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_inplace_merge_warp_monotonous()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, 2 * block_size> data;
	auto middle = data.begin() + block_size;

	gpu::enumerate(block, data.begin(), middle, [](int& value, gpu::offset_t id) {
		value = 2 * id;
	});
	gpu::enumerate(block, middle, data.end(), [](int& value, gpu::offset_t id) {
		value = id;
	});
	block.sync();

	if (block.thread_rank() < warp.size())
		gpu::inplace_merge(warp, data.begin(), middle, data.end());
	block.sync();

	ENSURE(gpu::is_sorted(block, data.begin(), data.end()));
}

TEST_CASE("INPLACE_MERGE", "[INPLACE_MERGE][ALGORITHM]")
{
	SECTION("BLOCK full array")
	{
		CHECK(launch(test_inplace_merge_block_full_array<40>));
		CHECK(launch(test_inplace_merge_block_full_array<256>));
		CHECK(launch(test_inplace_merge_block_full_array<1000>));
		CHECK(launch(test_inplace_merge_block_full_array<2048>));
	}

	SECTION("WARP full array")
	{
		CHECK(launch(test_inplace_merge_warp_full_array<10>));
		CHECK(launch(test_inplace_merge_warp_full_array<64>));
		CHECK(launch(test_inplace_merge_warp_full_array<200>));
		CHECK(launch(test_inplace_merge_warp_full_array<256>));
	}

	SECTION("BLOCK monotonous")
	{
		CHECK(launch(test_inplace_merge_block_monotonous<40>));
		CHECK(launch(test_inplace_merge_block_monotonous<256>));
		CHECK(launch(test_inplace_merge_block_monotonous<1000>));
		CHECK(launch(test_inplace_merge_block_monotonous<2048>));
	}

	SECTION("WARP monotonous")
	{
		CHECK(launch(test_inplace_merge_warp_monotonous<10>));
		CHECK(launch(test_inplace_merge_warp_monotonous<64>));
		CHECK(launch(test_inplace_merge_warp_monotonous<200>));
		CHECK(launch(test_inplace_merge_warp_monotonous<256>));
	}
}
