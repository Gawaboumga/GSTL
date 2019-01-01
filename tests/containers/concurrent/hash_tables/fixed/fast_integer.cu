#include <gstl/containers/concurrent/hash_tables/fixed/fast_integer.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

using allocator_type = gpu::linear_allocator<gpu::concurrent::default_layout_type<int, int>>;
using fast_integer_type = gpu::concurrent::fast_integer<int, int>;
using value_type = typename fast_integer_type::value_type;
using size_type = typename fast_integer_type::size_type;

constexpr static const size_type CAPACITY = 1 << 15;

GPU_DEVICE void build(gpu::block_t block, allocator_type* allocator, fast_integer_type* hash_table, size_type capacity)
{
	if (block.thread_rank() == 0)
		new (allocator) allocator_type(get_memory());
	block.sync();

	new (hash_table) fast_integer_type(block, *allocator, capacity);
	block.sync();
}

GPU_GLOBAL void test_concurrent_fast_integer_thread()
{
	GPU_SHARED allocator_type allocator;
	GPU_SHARED fast_integer_type hash_table;

	gpu::block_t block = gpu::this_thread_block();
	size_type capacity = CAPACITY;
	build(block, &allocator, &hash_table, capacity);

	int offset = 0;
	auto thid = block.thread_rank();
	while (offset + thid < capacity / 2)
	{
		auto first = offset + thid;
		auto second = thid;
		auto data = gpu::make_pair(first, second);
		auto it = hash_table.insert(data);
		ENSURE(it->first == first);
		ENSURE(it->second == second);

		auto found_it = hash_table.find(first);
		ENSURE(found_it->first == first);
		ENSURE(found_it->second == second);

		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.size() == capacity / 2);
}

GPU_GLOBAL void test_concurrent_fast_integer_collision_thread()
{
	GPU_SHARED allocator_type allocator;
	GPU_SHARED fast_integer_type hash_table;

	gpu::block_t block = gpu::this_thread_block();
	size_type capacity = CAPACITY;
	build(block, &allocator, &hash_table, capacity);

	int offset = 0;
	auto thid = block.thread_rank();
	while (offset + thid < capacity / 2)
	{
		auto it = hash_table.insert(gpu::make_pair(0, offset + thid));
		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.size() == 1);
}

GPU_GLOBAL void test_concurrent_fast_integer_erase_thread()
{
	GPU_SHARED allocator_type allocator;
	GPU_SHARED fast_integer_type hash_table;

	gpu::block_t block = gpu::this_thread_block();
	size_type capacity = CAPACITY;
	build(block, &allocator, &hash_table, capacity);

	int offset = 0;
	auto thid = block.thread_rank();
	while (offset + thid < capacity / 2)
	{
		auto it = hash_table.insert(gpu::make_pair(offset + thid, thid));
		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.size() == capacity / 2);

	offset = 0;
	while (offset + thid < capacity / 2)
	{
		hash_table.erase(offset + thid);
		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.empty());
}

GPU_GLOBAL void test_concurrent_fast_integer_tile()
{
	GPU_SHARED allocator_type allocator;
	GPU_SHARED fast_integer_type hash_table;

	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> tile = gpu::tiled_partition<32>(block);
	size_type capacity = CAPACITY / tile.size();
	build(block, &allocator, &hash_table, capacity);

	int offset = 0;
	auto thid = block.thread_rank();
	while (offset + thid < capacity / 2)
	{
		auto first = (offset + thid) / tile.size();
		auto second = thid / tile.size();
		auto data = gpu::make_pair(first, second);
		auto it = hash_table.insert(tile, data);
		ENSURE(it->first == first);
		ENSURE(it->second == second);

		auto found_it = hash_table.find(tile, first);
		ENSURE(found_it->first == first);
		ENSURE(found_it->second == second);

		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.size() == capacity / (2 * tile.size()));
}

GPU_GLOBAL void test_concurrent_fast_integer_collision_tile()
{
	GPU_SHARED allocator_type allocator;
	GPU_SHARED fast_integer_type hash_table;

	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> tile = gpu::tiled_partition<32>(block);
	size_type capacity = CAPACITY / tile.size();
	build(block, &allocator, &hash_table, capacity);

	int offset = 0;
	auto thid = block.thread_rank();
	while (offset + thid < capacity / 2)
	{
		auto it = hash_table.insert(tile, gpu::make_pair(0, offset + thid));
		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.size() == 1);
}

GPU_GLOBAL void test_concurrent_fast_integer_erase_tile()
{
	GPU_SHARED allocator_type allocator;
	GPU_SHARED fast_integer_type hash_table;

	gpu::block_t block = gpu::this_thread_block();
	gpu::block_tile_t<32> tile = gpu::tiled_partition<32>(block);
	size_type capacity = CAPACITY / tile.size();
	build(block, &allocator, &hash_table, capacity);

	int offset = 0;
	auto thid = block.thread_rank();
	while (offset + thid < capacity / 2)
	{
		auto it = hash_table.insert(gpu::make_pair((offset + thid) / tile.size(), thid));
		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.size() == capacity / (2 * tile.size()));

	offset = 0;
	while (offset + thid < capacity / 2)
	{
		hash_table.erase(tile, (offset + thid) / tile.size());
		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.empty());
}

TEST_CASE("FAST_INTEGER", "[FAST_INTEGER][CONTAINERS][CONCURRENT][HASH_TABLES]")
{
	SECTION("Fast integer per thread")
	{
		CHECK(launch(test_concurrent_fast_integer_thread));
		CHECK(launch(test_concurrent_fast_integer_collision_thread));
		CHECK(launch(test_concurrent_fast_integer_erase_thread));
	}

	SECTION("Fast integer per tile")
	{
		CHECK(launch(test_concurrent_fast_integer_tile));
		CHECK(launch(test_concurrent_fast_integer_collision_tile));
		CHECK(launch(test_concurrent_fast_integer_erase_tile));
	}
}
