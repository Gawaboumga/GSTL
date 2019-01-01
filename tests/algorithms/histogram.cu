#include <gstl/algorithms/histogram.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/enumerate.cuh>
#include <gstl/algorithms/equal.cuh>
#include <gstl/algorithms/fill.cuh>
#include <gstl/algorithms/minmax.cuh>
#include <gstl/containers/array.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_histogram_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> in;
	constexpr const unsigned int HISTOGRAM_SIZE = 128;
	GPU_SHARED gpu::array<int, HISTOGRAM_SIZE> out;

	gpu::fill(block, out.begin(), out.end(), 0);
	gpu::enumerate(block, in.begin(), in.end(), [](int& v, gpu::offset_t index) {
		v = index % out.size();
	});
	block.sync();

	gpu::histogram(block, in.begin(), in.end(), out.begin(), [](int v) {
		return v % out.size();
	});
	block.sync();

	gpu::enumerate(block, out.begin(), out.end(), [](int v, gpu::offset_t index) {
		gpu::offset_t number_of_times = in.size() / out.size();
		ENSURE(v == index < out.size() ? number_of_times + 1 : number_of_times);
	});
}

template <unsigned int block_size>
GPU_GLOBAL void test_histogram_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> in;
	constexpr const unsigned int HISTOGRAM_SIZE = 128;
	GPU_SHARED gpu::array<int, HISTOGRAM_SIZE> out;

	gpu::fill(block, out.begin(), out.end(), 0);
	gpu::enumerate(block, in.begin(), in.end(), [](int& v, gpu::offset_t index) {
		v = index % out.size();
	});
	block.sync();

	if (block.thread_rank() < warp.size())
	{
		gpu::histogram(warp, in.begin(), in.end(), out.begin(), [](int v) {
			return v % out.size();
		});
	}
	block.sync();

	gpu::enumerate(block, out.begin(), out.end(), [](int v, gpu::offset_t index) {
		gpu::offset_t number_of_times = in.size() / out.size();
		ENSURE(v == index < out.size() ? number_of_times + 1 : number_of_times);
	});
}

TEST_CASE("HISTOGRAM", "[HISTOGRAM][ALGORITHM]")
{
	SECTION("BLOCK histogram")
	{
		CHECK(launch(test_histogram_block<200>));
		CHECK(launch(test_histogram_block<256>));
		CHECK(launch(test_histogram_block<2000>));
		CHECK(launch(test_histogram_block<2048>));
	}

	SECTION("WARP histogram")
	{
		CHECK(launch(test_histogram_warp<200>));
		CHECK(launch(test_histogram_warp<256>));
		CHECK(launch(test_histogram_warp<2000>));
		CHECK(launch(test_histogram_warp<2048>));
	}
}
