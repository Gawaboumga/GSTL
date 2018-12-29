#include <gstl/containers/concurrent/hash_tables/fixed/fast_integer.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

GPU_GLOBAL void test_concurrent_fast_integer_thread()
{
	using allocator_type = gpu::linear_allocator<gpu::concurrent::default_layout_type<int, int>>;
	using fast_integer_type = gpu::concurrent::fast_integer<int, int>;
	using value_type = typename fast_integer_type::value_type;

	GPU_SHARED allocator_type allocator;

	gpu::block_t block = gpu::this_thread_block();
	if (block.thread_rank() == 0)
		new (&allocator) allocator_type(get_memory());
	block.sync();

	GPU_SHARED fast_integer_type hash_table;

	typename fast_integer_type::size_type capacity = 1 << 7;
	new (&hash_table) fast_integer_type(block, allocator, capacity);
	block.sync();

	int offset = 0;
	auto thid = block.thread_rank();
	while (offset + thid < capacity)
	{
		auto first = offset + thid;
		auto second = thid;
		auto data = gpu::make_pair(first, second);
		auto it = hash_table.insert(data);
		ENSURE(it->first == first);
		ENSURE(it->second == second);

		offset += block.size();
	}
	block.sync();
	ENSURE(hash_table.size() == capacity);
}

TEST_CASE("FAST_INTEGER", "[FAST_INTEGER][CONTAINERS][CONCURRENT][HASH_TABLES]")
{
	SECTION("Fast integer per thread")
	{
		CHECK(launch(test_concurrent_fast_integer_thread));
	}
}
