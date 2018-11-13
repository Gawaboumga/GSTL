#include <gstl/algorithms/transform.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/equal.cuh>
#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_transform_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> in;
	GPU_SHARED gpu::array<int, block_size> out;
	GPU_SHARED gpu::array<int, block_size> theoretical_output;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	gpu::fill(block, theoretical_output.begin(), theoretical_output.end(), 2 * value);

	block.sync();

	gpu::transform(block, in.begin(), in.end(), out.begin(), [](int v) {
		return 2 * v;
	});

	block.sync();

	ENSURE(gpu::equal(block, out.begin(), out.end(), theoretical_output.begin()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_transform_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> in;
	GPU_SHARED gpu::array<int, block_size> out;
	GPU_SHARED gpu::array<int, block_size> theoretical_output;

	int value = 5;
	gpu::fill(block, in.begin(), in.end(), value);
	gpu::fill(block, theoretical_output.begin(), theoretical_output.end(), 2 * value);

	block.sync();

	if (block.thread_rank() < warp.size())
	{
		gpu::transform(block, in.begin(), in.end(), out.begin(), [](int v) {
			return 2 * v;
		});
	}

	block.sync();

	ENSURE(gpu::equal(block, out.begin(), out.end(), theoretical_output.begin()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_transform_binary_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> in1;
	GPU_SHARED gpu::array<int, block_size> in2;
	GPU_SHARED gpu::array<int, block_size> out;
	GPU_SHARED gpu::array<int, block_size> theoretical_output;

	int value = 5;
	gpu::fill(block, in1.begin(), in1.end(), value);
	gpu::fill(block, in2.begin(), in2.end(), value);
	gpu::fill(block, theoretical_output.begin(), theoretical_output.end(), 3 * value);

	block.sync();

	gpu::transform(block, in1.begin(), in1.end(), in2.begin(), out.begin(), [](int x, int y) {
		return 2 * x + y;
	});

	block.sync();

	ENSURE(gpu::equal(block, out.begin(), out.end(), theoretical_output.begin()));
}

template <unsigned int block_size>
GPU_GLOBAL void test_transform_binary_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> in1;
	GPU_SHARED gpu::array<int, block_size> in2;
	GPU_SHARED gpu::array<int, block_size> out;
	GPU_SHARED gpu::array<int, block_size> theoretical_output;

	int value = 5;
	gpu::fill(block, in1.begin(), in1.end(), value);
	gpu::fill(block, in2.begin(), in2.end(), value);
	gpu::fill(block, theoretical_output.begin(), theoretical_output.end(), 3 * value);

	block.sync();

	if (block.thread_rank() < warp.size())
	{
		gpu::transform(block, in1.begin(), in1.end(), in2.begin(), out.begin(), [](int x, int y) {
			return 2 * x + y;
		});
	}

	block.sync();

	ENSURE(gpu::equal(block, out.begin(), out.end(), theoretical_output.begin()));
}

TEST_CASE("TRANSFORM", "[TRANSFORM][ALGORITHM]")
{
	SECTION("BLOCK transform")
	{
		CHECK(launch(test_transform_block<200>));
		CHECK(launch(test_transform_block<256>));
		CHECK(launch(test_transform_block<2000>));
		CHECK(launch(test_transform_block<2048>));
	}

	SECTION("WARP transform")
	{
		CHECK(launch(test_transform_warp<200>));
		CHECK(launch(test_transform_warp<256>));
		CHECK(launch(test_transform_warp<2000>));
		CHECK(launch(test_transform_warp<2048>));
	}

	SECTION("BLOCK transform binary")
	{
		CHECK(launch(test_transform_binary_block<200>));
		CHECK(launch(test_transform_binary_block<256>));
		CHECK(launch(test_transform_binary_block<2000>));
		CHECK(launch(test_transform_binary_block<2048>));
	}

	SECTION("WARP transform binary")
	{
		CHECK(launch(test_transform_binary_warp<200>));
		CHECK(launch(test_transform_binary_warp<256>));
		CHECK(launch(test_transform_binary_warp<2000>));
		CHECK(launch(test_transform_binary_warp<2048>));
	}
}
