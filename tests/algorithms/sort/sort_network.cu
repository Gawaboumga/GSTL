#include <gstl/algorithms/sort/bitonic_sort.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/copy.cuh>
#include <gstl/algorithms/generate.cuh>
#include <gstl/algorithms/is_sorted.cuh>
#include <gstl/containers/array.cuh>
#include <gstl/random/linear_congruential_engine.cuh>
#include <gstl/random/random_device.cuh>
#include <gstl/random/uniform_int_distribution.cuh>

template <unsigned int block_size>
GPU_DEVICE void test_sort_network_generate_random_number(gpu::block_t block, gpu::array<int, block_size>& data)
{
	gpu::random_device rd;
	gpu::minstd_rand gen(rd());
	gpu::uniform_int_distribution<> dis(1, 6);

	gpu::generate(block, data.begin(), data.end(), [&dis, &gen]() {
		return dis(gen);
	});
	block.sync();
}

template <unsigned int block_size>
GPU_GLOBAL void test_sort_network_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> data;
	GPU_SHARED gpu::array<int, block_size> copy;

	test_sort_network_generate_random_number(block, data);
	gpu::copy(data.begin(), data.end(), copy.begin());
	block.sync();

	gpu::bitonic_sort(block, copy.begin(), copy.end());
	block.sync();

	ENSURE(gpu::is_sorted(block, copy.begin(), copy.end()));
	block.sync();
}

template <unsigned int block_size>
GPU_GLOBAL void test_sort_network_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> data;
	GPU_SHARED gpu::array<int, block_size> copy;

	test_sort_network_generate_random_number(block, data);
	gpu::copy(data.begin(), data.end(), copy.begin());
	block.sync();

	if (block.thread_rank() < warp.size())
		gpu::bitonic_sort(warp, copy.begin(), copy.end());
	block.sync();

	ENSURE(gpu::is_sorted(block, copy.begin(), copy.end()));
	block.sync();
}

TEST_CASE("SORT_NETWORK", "[SORT_NETWORK][ALGORITHM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_sort_network_block<200>));
		CHECK(launch(test_sort_network_block<256>));
		CHECK(launch(test_sort_network_block<2000>));
		CHECK(launch(test_sort_network_block<2048>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_sort_network_warp<200>));
		CHECK(launch(test_sort_network_warp<256>));
		CHECK(launch(test_sort_network_warp<2000>));
		CHECK(launch(test_sort_network_warp<2048>));
	}
}
