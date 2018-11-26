#include <gstl/algorithms/merge.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/equal.cuh>
#include <gstl/algorithms/enumerate.cuh>
#include <gstl/algorithms/fill.cuh>
#include <gstl/algorithms/is_sorted.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_merge_block_full_array()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> in1;
	GPU_SHARED gpu::array<int, block_size> in2;
	GPU_SHARED gpu::array<int, 2 * block_size> output;

	int value = 5;
	gpu::fill(block, in1.begin(), in1.end(), value);
	gpu::fill(block, in2.begin(), in2.end(), value + 1);
	block.sync();

	gpu::merge(block, in1.begin(), in1.end(), in2.begin(), in2.end(), output.begin());
	block.sync();

	ENSURE(gpu::equal(block, in1.begin(), in1.end(), output.begin()));
	ENSURE(gpu::equal(block, in2.begin(), in2.end(), output.begin() + block_size));
}

template <unsigned int block_size>
GPU_GLOBAL void test_merge_warp_full_array()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> in1;
	GPU_SHARED gpu::array<int, block_size> in2;
	GPU_SHARED gpu::array<int, 2 * block_size> output;

	int value = 5;
	gpu::fill(block, in1.begin(), in1.end(), value);
	gpu::fill(block, in2.begin(), in2.end(), value + 1);
	block.sync();

	if (block.thread_rank() < warp.size())
		gpu::merge(warp, in1.begin(), in1.end(), in2.begin(), in2.end(), output.begin());
	block.sync();

	ENSURE(gpu::equal(block, in1.begin(), in1.end(), output.begin()));
	ENSURE(gpu::equal(block, in2.begin(), in2.end(), output.begin() + block_size));
}

template <unsigned int block_size>
GPU_GLOBAL void test_merge_block_monotonous()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> in1;
	GPU_SHARED gpu::array<int, block_size> in2;
	GPU_SHARED gpu::array<int, 2 * block_size> output;

	gpu::enumerate(block, in1.begin(), in1.end(), [](int& value, gpu::offset_t id) {
		value = 2 * id;
	});
	gpu::enumerate(block, in2.begin(), in2.end(), [](int& value, gpu::offset_t id) {
		value = id;
	});
	block.sync();

	gpu::merge(block, in1.begin(), in1.end(), in2.begin(), in2.end(), output.begin());
	block.sync();

	ENSURE(gpu::is_sorted(block, in1.begin(), in1.end()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_merge_warp_monotonous()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> in1;
	GPU_SHARED gpu::array<int, block_size> in2;
	GPU_SHARED gpu::array<int, 2 * block_size> output;

	gpu::enumerate(block, in1.begin(), in1.end(), [](int& value, gpu::offset_t id) {
		value = 2 * id;
	});
	gpu::enumerate(block, in2.begin(), in2.end(), [](int& value, gpu::offset_t id) {
		value = id;
	});
	block.sync();

	if (block.thread_rank() < warp.size())
		gpu::merge(warp, in1.begin(), in1.end(), in2.begin(), in2.end(), output.begin());
	block.sync();

	ENSURE(gpu::is_sorted(block, in1.begin(), in1.end()));
}

TEST_CASE("MERGE", "[MERGE][ALGORITHM]")
{
	SECTION("BLOCK full array")
	{
		CHECK(launch(test_merge_block_full_array<200>));
		CHECK(launch(test_merge_block_full_array<256>));
		CHECK(launch(test_merge_block_full_array<2000>));
		CHECK(launch(test_merge_block_full_array<2048>));
	}

	SECTION("WARP full array")
	{
		CHECK(launch(test_merge_warp_full_array<200>));
		CHECK(launch(test_merge_warp_full_array<256>));
		CHECK(launch(test_merge_warp_full_array<2000>));
		CHECK(launch(test_merge_warp_full_array<2048>));
	}

	SECTION("BLOCK monotonous")
	{
		CHECK(launch(test_merge_block_monotonous<200>));
		CHECK(launch(test_merge_block_monotonous<256>));
		CHECK(launch(test_merge_block_monotonous<2000>));
		CHECK(launch(test_merge_block_monotonous<2048>));
	}

	SECTION("WARP monotonous")
	{
		CHECK(launch(test_merge_warp_monotonous<200>));
		CHECK(launch(test_merge_warp_monotonous<256>));
		CHECK(launch(test_merge_warp_monotonous<2000>));
		CHECK(launch(test_merge_warp_monotonous<2048>));
	}
}
