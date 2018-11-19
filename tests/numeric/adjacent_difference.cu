#include <gstl/numeric/adjacent_difference.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/algorithms/sort.cuh>
#include <gstl/containers/array.cuh>
#include <gstl/functional/function_object.cuh>

template <unsigned int number_of_elements>
GPU_GLOBAL void test_adjacent_difference_block()
{
	gpu::block_t block = gpu::this_thread_block();
	GPU_SHARED gpu::array<int, number_of_elements> input;
	GPU_SHARED gpu::array<int, number_of_elements> output;

	gpu::fill(block, input.begin(), input.end(), 1);
	block.sync();

	auto end_ptr = gpu::adjacent_difference(block, input.begin(), input.end(), output.begin());
	block.sync();

	ENSURE(end_ptr == output.end());
	ENSURE(*output.begin() == 1);
	ENSURE(*(--end_ptr) == 0);
}

template <unsigned int number_of_elements>
GPU_GLOBAL void test_adjacent_difference_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	GPU_SHARED gpu::array<int, number_of_elements> input;
	GPU_SHARED gpu::array<int, number_of_elements> output;

	gpu::fill(block, input.begin(), input.end(), 1);
	block.sync();

	if (block.thread_rank() < warp.size())
	{
		auto end_ptr = gpu::adjacent_difference(warp, input.begin(), input.end(), output.begin());
		ENSURE(end_ptr == output.end());
		ENSURE(*output.begin() == 1);
		ENSURE(*(--end_ptr) == 0);
	}
}

TEST_CASE("ADJACENT DIFFERENCE", "[ADJACENT_DIFFERENCE][NUMERIC]")
{
	SECTION("BLOCK adjacent_difference")
	{
		CHECK(launch(test_adjacent_difference_block<200>));
		CHECK(launch(test_adjacent_difference_block<256>));
		CHECK(launch(test_adjacent_difference_block<2000>));
		CHECK(launch(test_adjacent_difference_block<2048>));
	}

	SECTION("WARP exclusive_scan")
	{
		CHECK(launch(test_adjacent_difference_warp<200>));
		CHECK(launch(test_adjacent_difference_warp<256>));
		CHECK(launch(test_adjacent_difference_warp<2000>));
		CHECK(launch(test_adjacent_difference_warp<2048>));
	}
}
