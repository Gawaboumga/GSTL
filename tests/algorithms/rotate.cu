#include <gstl/algorithms/rotate.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/binary_search.cuh>
#include <gstl/algorithms/enumerate.cuh>
#include <gstl/algorithms/equal.cuh>
#include <gstl/algorithms/is_sorted.cuh>
#include <gstl/containers/array.cuh>
#include <gstl/random/sample.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_rotate_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> data;
	gpu::randint(block, data.begin(), data.size(), 0, 256);
	block.size();

	for (auto it = data.begin(); it != data.end(); ++it)
	{
		gpu::rotate(block, gpu::upper_bound(block, data.begin(), it, *it), it, it + 1);
		block.sync();
	}

	ENSURE(gpu::is_sorted(block, data.begin(), data.end()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_rotate_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> data;
	gpu::randint(block, data.begin(), data.size(), 0, 256);
	block.size();

	if (block.thread_rank() < warp.size())
	{
		for (auto it = data.begin(); it != data.end(); ++it)
		{
			gpu::rotate(warp, gpu::upper_bound(warp, data.begin(), it, *it), it, it + 1);
			warp.sync();
		}
	}
	block.sync();

	ENSURE(gpu::is_sorted(block, data.begin(), data.end()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_rotate_inside_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> data;
	GPU_SHARED gpu::array<int, block_size> result;
	gpu::enumerate(block, data.begin(), data.end(), [](int& value, gpu::offset_t offset) {
		value = offset;
	});
	block.sync();

	auto end_it = data.end();
	gpu::offset_t relative_offset = 3;
	gpu::offset_t middle_point = (block_size * 3) / 4;
	gpu::rotate(block, data.begin() + relative_offset, data.begin() + middle_point, end_it - relative_offset);
	block.sync();

	gpu::enumerate(block, result.begin(), result.begin() + relative_offset, [](int& value, gpu::offset_t offset) {
		value = offset;
	});
	gpu::enumerate(block, result.begin() + relative_offset, result.begin() + data.size() - middle_point, [middle_point](int& value, gpu::offset_t offset) {
		value = middle_point + offset;
	});
	gpu::enumerate(block, result.begin() + (data.size() - middle_point), result.end() - relative_offset, [relative_offset](int& value, gpu::offset_t offset) {
		value = relative_offset + offset;
	});
	gpu::enumerate(block, result.end() - relative_offset, result.end(), [relative_offset](int& value, gpu::offset_t offset) {
		value = data.size() - relative_offset + offset;
	});
	block.sync();
	ENSURE(gpu::equal(block, data.begin(), data.end(), result.begin()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_rotate_inside_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> data;
	GPU_SHARED gpu::array<int, block_size> result;
	gpu::enumerate(block, data.begin(), data.end(), [](int& value, gpu::offset_t offset) {
		value = offset;
	});
	block.sync();

	auto end_it = data.end();
	gpu::offset_t relative_offset = 3;
	gpu::offset_t middle_point = (block_size * 3) / 4;

	if (block.thread_rank() < warp.size())
		gpu::rotate(warp, data.begin() + relative_offset, data.begin() + middle_point, end_it - relative_offset);
	block.sync();

	gpu::enumerate(block, result.begin(), result.begin() + relative_offset, [](int& value, gpu::offset_t offset) {
		value = offset;
	});
	gpu::enumerate(block, result.begin() + relative_offset, result.begin() + data.size() - middle_point, [middle_point](int& value, gpu::offset_t offset) {
		value = middle_point + offset;
	});
	gpu::enumerate(block, result.begin() + (data.size() - middle_point), result.end() - relative_offset, [relative_offset](int& value, gpu::offset_t offset) {
		value = relative_offset + offset;
	});
	gpu::enumerate(block, result.end() - relative_offset, result.end(), [relative_offset](int& value, gpu::offset_t offset) {
		value = data.size() - relative_offset + offset;
	});
	block.sync();
	ENSURE(gpu::equal(block, data.begin(), data.end(), result.begin()));
}

TEST_CASE("rotate", "[ROTATE][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_rotate_block<200>));
		CHECK(launch(test_rotate_block<256>));
		CHECK(launch(test_rotate_block<1000>));
		CHECK(launch(test_rotate_block<1024>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_rotate_warp<20>));
		CHECK(launch(test_rotate_warp<128>));
		CHECK(launch(test_rotate_warp<200>));
		CHECK(launch(test_rotate_warp<256>));
	}

	SECTION("Classical rotation by block")
	{
		CHECK(launch(test_rotate_inside_block<200>));
		CHECK(launch(test_rotate_inside_block<256>));
		CHECK(launch(test_rotate_inside_block<1000>));
		CHECK(launch(test_rotate_inside_block<1024>));
	}

	SECTION("Classical rotation by warp")
	{
		CHECK(launch(test_rotate_warp<20>));
		CHECK(launch(test_rotate_warp<128>));
		CHECK(launch(test_rotate_warp<200>));
		CHECK(launch(test_rotate_warp<256>));
	}
}
