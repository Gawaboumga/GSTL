#include <gstl/numeric/inner_product.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>
#include <gstl/functional/function_object.cuh>

template <unsigned int number_of_elements>
GPU_GLOBAL void test_inner_product_block()
{
	gpu::block_t block = gpu::this_thread_block();
	GPU_SHARED gpu::array<int, number_of_elements> data;

	gpu::fill(block, data.begin(), data.end(), 1);
	block.sync();

	gpu::group_result<int> g_result = gpu::inner_product(block, data.begin(), data.end(), data.begin(), 0);
	int result = g_result.broadcast(block);
	ENSURE(result == number_of_elements);
}

template <unsigned int number_of_elements>
GPU_GLOBAL void test_inner_product_block_two_arrays()
{
	gpu::block_t block = gpu::this_thread_block();
	GPU_SHARED gpu::array<int, number_of_elements> data1;
	GPU_SHARED gpu::array<int, number_of_elements> data2;

	gpu::fill(block, data1.begin(), data1.end(), 1);
	gpu::fill(block, data2.begin(), data2.end(), 2);
	block.sync();

	gpu::group_result<int> g_result = gpu::inner_product(block, data1.begin(), data1.end(), data2.begin(), 0);
	int result = g_result.broadcast(block);
	ENSURE(result == 2 * number_of_elements);
}

template <unsigned int number_of_elements>
GPU_GLOBAL void test_inner_product_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	GPU_SHARED gpu::array<int, number_of_elements> data;

	gpu::fill(block, data.begin(), data.end(), 1);
	block.sync();

	if (block.thread_rank() < warp.size())
	{
		gpu::group_result<int> g_result = gpu::inner_product(warp, data.begin(), data.end(), data.begin(), 0);
		int result = g_result.broadcast(warp);
		ENSURE(result == number_of_elements);
	}
}

template <unsigned int number_of_elements>
GPU_GLOBAL void test_inner_product_warp_two_arrays()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);
	GPU_SHARED gpu::array<int, number_of_elements> data1;
	GPU_SHARED gpu::array<int, number_of_elements> data2;

	gpu::fill(block, data1.begin(), data1.end(), 1);
	gpu::fill(block, data2.begin(), data2.end(), 2);
	block.sync();

	if (block.thread_rank() < warp.size())
	{
		gpu::group_result<int> g_result = gpu::inner_product(warp, data1.begin(), data1.end(), data2.begin(), 0);
		int result = g_result.broadcast(warp);
		ENSURE(result == 2 * number_of_elements);
	}
}

TEST_CASE("INNER PRODUCT", "[INNER_PRODUCT][NUMERIC]")
{
	SECTION("BLOCK inner_product")
	{
		CHECK(launch(test_inner_product_block<200>));
		CHECK(launch(test_inner_product_block<256>));
		CHECK(launch(test_inner_product_block<2000>));
		CHECK(launch(test_inner_product_block<2048>));
	}

	SECTION("WARP inner_product")
	{
		CHECK(launch(test_inner_product_warp<200>));
		CHECK(launch(test_inner_product_warp<256>));
		CHECK(launch(test_inner_product_warp<2000>));
		CHECK(launch(test_inner_product_warp<2048>));
	}

	SECTION("BLOCK inner_product on two arrays")
	{
		CHECK(launch(test_inner_product_block_two_arrays<200>));
		CHECK(launch(test_inner_product_block_two_arrays<256>));
		CHECK(launch(test_inner_product_block_two_arrays<2000>));
		CHECK(launch(test_inner_product_block_two_arrays<2048>));
	}

	SECTION("WARP inner_product on two arrays")
	{
		CHECK(launch(test_inner_product_warp_two_arrays<200>));
		CHECK(launch(test_inner_product_warp_two_arrays<256>));
		CHECK(launch(test_inner_product_warp_two_arrays<2000>));
		CHECK(launch(test_inner_product_warp_two_arrays<2048>));
	}
}
