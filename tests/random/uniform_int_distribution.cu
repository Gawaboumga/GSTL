#include <gstl/random/uniform_int_distribution.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/generate.cuh>
#include <gstl/containers/array.cuh>
#include <gstl/random/linear_congruential_engine.cuh>
#include <gstl/random/random_device.cuh>

template <unsigned int block_size>
GPU_GLOBAL void test_uniform_int_distribution_block()
{
	gpu::block_t block = gpu::this_thread_block();

	GPU_SHARED gpu::array<int, block_size> data;

	gpu::random_device rd;
	gpu::minstd_rand gen(rd());
	gpu::uniform_int_distribution<> dis(1, 6);

	gpu::generate(block, data.begin(), data.end(), [&dis, &gen]() {
		return dis(gen);
	});
	block.sync();
}

template <unsigned int block_size>
GPU_GLOBAL void test_uniform_int_distribution_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	GPU_SHARED gpu::array<int, block_size> data;

	gpu::random_device rd;
	gpu::minstd_rand gen(rd());
	gpu::uniform_int_distribution<> dis(1, 6);

	if (block.thread_rank() < warp.size())
	{
		gpu::generate(warp, data.begin(), data.end(), [&dis, &gen]() {
			return dis(gen);
		});
	}
	block.sync();

	if (block.thread_rank() == 0)
	{
		for (auto&& m : data)
			printf("%d, ", m);
	}
}

TEST_CASE("UNIFORM_INT_DISTRIBUTION", "[UNIFORM_INT_DISTRIBUTION][RANDOM]")
{
	SECTION("BLOCK")
	{
		CHECK(launch(test_uniform_int_distribution_block<200>));
		CHECK(launch(test_uniform_int_distribution_block<256>));
		CHECK(launch(test_uniform_int_distribution_block<2000>));
		CHECK(launch(test_uniform_int_distribution_block<2048>));
	}

	SECTION("WARP")
	{
		CHECK(launch(test_uniform_int_distribution_warp<200>));
		CHECK(launch(test_uniform_int_distribution_warp<256>));
		CHECK(launch(test_uniform_int_distribution_warp<2000>));
		CHECK(launch(test_uniform_int_distribution_warp<2048>));
	}
}
