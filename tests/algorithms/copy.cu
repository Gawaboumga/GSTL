#include <gstl/algorithms/copy.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/equal.cuh>
#include <gstl/algorithms/enumerate.cuh>
#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_copy_if_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size * 2> in;
	GPU_SHARED gpu::array<int, block_size> out;
	GPU_SHARED gpu::array<int, block_size> theoretical_output;

	gpu::enumerate(block, in.begin(), in.end(), [](int& value, gpu::offset_t id) {
		value = id;
	});
	gpu::enumerate(block, theoretical_output.begin(), theoretical_output.end(), [](int& value, gpu::offset_t id) {
		value = id * 2;
	});

	block.sync();

	gpu::copy_if(block, in.begin(), in.end(), out.begin(), [](int v) {
		return v % 2;
	});

	block.sync();

	ENSURE(gpu::equal(block, out.begin(), out.end(), theoretical_output.begin()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_copy_if_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size * 2> in;
	GPU_SHARED gpu::array<int, block_size> out;
	GPU_SHARED gpu::array<int, block_size> theoretical_output;

	gpu::enumerate(block, in.begin(), in.end(), [](int& value, gpu::offset_t id) {
		value = id;
	});
	gpu::enumerate(block, theoretical_output.begin(), theoretical_output.end(), [](int& value, gpu::offset_t id) {
		value = id * 2;
	});

	block.sync();

	if (block.thread_rank() < warp.thread_rank())
	{
		gpu::copy_if(block, in.begin(), in.end(), out.begin(), [](int v) {
			return v % 2;
		});
	}
	warp.sync();

	ENSURE(gpu::equal(block, out.begin(), out.end(), theoretical_output.begin()));
}

TEST_CASE("COPY", "[COPY][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_copy_if_block<200>));
		CHECK(launch(test_copy_if_block<256>));
		CHECK(launch(test_copy_if_block<2000>));
		CHECK(launch(test_copy_if_block<2048>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_copy_if_warp<200>));
		CHECK(launch(test_copy_if_warp<256>));
		CHECK(launch(test_copy_if_warp<2000>));
		CHECK(launch(test_copy_if_warp<2048>));
	}
}
