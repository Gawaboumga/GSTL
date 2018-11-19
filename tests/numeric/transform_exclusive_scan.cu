#include <gstl/numeric/transform_exclusive_scan.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/algorithms/sort.cuh>
#include <gstl/containers/array.cuh>
#include <gstl/functional/function_object.cuh>

template <unsigned int number_of_elements>
GPU_GLOBAL void test_transform_exclusive_scan_block()
{
	gpu::block_t block = gpu::this_thread_block();
	GPU_SHARED gpu::array<int, number_of_elements> input;
	GPU_SHARED gpu::array<int, number_of_elements> output;

	gpu::fill(block, input.begin(), input.end(), 1);
	block.sync();

	auto end_ptr = gpu::transform_exclusive_scan(block, input.begin(), input.end(), output.begin(), gpu::plus<>(), gpu::identity());
	block.sync();

	ENSURE(end_ptr == output.end());
	ENSURE(gpu::is_sorted(output.begin(), output.end()));
	ENSURE(*output.begin() == 0);
	ENSURE(*(--end_ptr) == number_of_elements - 1);
}

template <unsigned int number_of_elements>
GPU_GLOBAL void test_transform_exclusive_scan_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	GPU_SHARED gpu::array<int, number_of_elements> input;
	GPU_SHARED gpu::array<int, number_of_elements> output;

	gpu::fill(block, input.begin(), input.end(), 1);
	block.sync();

	if (block.thread_rank() < warp.size())
	{
		auto end_ptr = gpu::transform_exclusive_scan(warp, input.begin(), input.end(), output.begin(), gpu::plus<>(), gpu::identity());

		ENSURE(end_ptr == output.end());
		ENSURE(gpu::is_sorted(output.begin(), output.end()));
		ENSURE(*output.begin() == 0);
		ENSURE(*(--end_ptr) == number_of_elements - 1);
	}
}

TEST_CASE("TRANSFORM EXCLUSIVE SCAN", "[TRANSFORM_EXCLUSIVE_SCAN][NUMERIC]")
{
	SECTION("BLOCK transform_exclusive_scan")
	{
		CHECK(launch(test_transform_exclusive_scan_block<200>));
		CHECK(launch(test_transform_exclusive_scan_block<256>));
		CHECK(launch(test_transform_exclusive_scan_block<2000>));
		CHECK(launch(test_transform_exclusive_scan_block<2048>));
	}

	SECTION("WARP transform_exclusive_scan")
	{
		CHECK(launch(test_transform_exclusive_scan_warp<200>));
		CHECK(launch(test_transform_exclusive_scan_warp<256>));
		CHECK(launch(test_transform_exclusive_scan_warp<2000>));
		CHECK(launch(test_transform_exclusive_scan_warp<2048>));
	}
}
