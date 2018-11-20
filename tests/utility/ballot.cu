#include <gstl/utility/ballot.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

template <bool full>
GPU_GLOBAL void test_all_block()
{
	gpu::block_t block = gpu::this_thread_block();

	if (full)
	{
		ENSURE(gpu::all(block, true));
	}
	else
		ENSURE(!(gpu::all(block, block.thread_rank() != 47)));
}

template <bool full>
GPU_GLOBAL void test_all_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	if (block.thread_rank() >= warp.size())
		return;

	if (full)
	{
		ENSURE(gpu::all(warp, true));
	}
	else
		ENSURE(!(gpu::all(warp, block.thread_rank() != 17)));
}

template <bool none>
GPU_GLOBAL void test_any_block()
{
	gpu::block_t block = gpu::this_thread_block();

	if (none)
	{
		ENSURE(!(gpu::any(block, false)));
	}
	else
		ENSURE(gpu::any(block, block.thread_rank() == 47));
}

template <bool none>
GPU_GLOBAL void test_any_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	if (block.thread_rank() >= warp.size())
		return;

	if (none)
	{
		ENSURE(!(gpu::any(warp, false)));
	}
	else
		ENSURE(gpu::any(warp, block.thread_rank() == 17));
}

GPU_GLOBAL void test_first_index_block()
{
	gpu::block_t block = gpu::this_thread_block();

	gpu::offset_t result = gpu::first_index(block, block.thread_rank() == 147, 0);
	ENSURE(result == 147);

	result = gpu::first_index(block, block.thread_rank() == 147 || block.thread_rank() == 149, 148);
	ENSURE(result == 149);
}

GPU_GLOBAL void test_first_index_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	if (block.thread_rank() >= warp.size())
		return;

	gpu::offset_t result = gpu::first_index(warp, false);
	ENSURE(result == warp.size());

	result = gpu::first_index(warp, block.thread_rank() == 17, 0);
	ENSURE(result == 17);

	result = gpu::first_index(warp, block.thread_rank() == 17 || block.thread_rank() == 19, 18);
	ENSURE(result == 19);
}

GPU_GLOBAL void test_shfl_block()
{
	gpu::block_t block = gpu::this_thread_block();

	gpu::offset_t result = gpu::shfl(block, block.thread_rank(), 147);
	ENSURE(result == 146);
}

GPU_GLOBAL void test_shfl_warp()
{
	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> warp = gpu::tiled_partition<32>(block);

	if (block.thread_rank() >= warp.size())
		return;

	gpu::offset_t result = gpu::shfl(warp, warp.thread_rank(), 17);
	ENSURE(result == 16);
}

TEST_CASE("BALLOT", "[BALLOT][UTILITY]")
{
	SECTION("ALL")
	{
		CHECK(launch(test_all_block<true>));
		CHECK(launch(test_all_warp<true>));
		CHECK(launch(test_all_block<false>));
		CHECK(launch(test_all_warp<false>));
	}

	SECTION("ANY")
	{
		CHECK(launch(test_any_block<true>));
		CHECK(launch(test_any_warp<true>));
		CHECK(launch(test_any_block<false>));
		CHECK(launch(test_any_warp<false>));
	}

	SECTION("FIRST_INDEX")
	{
		CHECK(launch(test_first_index_block));
		CHECK(launch(test_first_index_warp));
	}

	SECTION("SHFL")
	{
		CHECK(launch(test_shfl_block));
		CHECK(launch(test_shfl_warp));
	}
}
