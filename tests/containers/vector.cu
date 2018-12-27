#include <gstl/containers/vector.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

GPU_GLOBAL void test_vector_constructor_block()
{
	using allocator_type = gpu::linear_allocator<int>;
	using vector_type = gpu::vector<int, allocator_type>;
	using size_type = typename vector_type::size_type;

	GPU_SHARED gpu::array<char, sizeof(int) * 600> local_memory;
	GPU_SHARED allocator_type alloc;

	gpu::block_t block = gpu::this_thread_block();
	if (block.thread_rank() == 0)
		new (&alloc) allocator_type(local_memory.data(), local_memory.size());
	block.sync();

	{
		GPU_SHARED vector_type vec;
		if (block.thread_rank() == 0)
			new (&vec) vector_type(alloc);
		block.sync();
		ENSURE(vec.capacity() == 0);
		ENSURE(vec.empty());
		ENSURE(vec.size() == 0);
	}

	{
		size_type count = 7;
		GPU_SHARED vector_type vec;
		new (&vec) vector_type(block, count, 5, alloc);
		ENSURE(vec.capacity() >= count);
		ENSURE(!vec.empty());
		ENSURE(vec.size() == count);
	}
}

GPU_GLOBAL void test_vector_push_back_block()
{
	using allocator_type = gpu::linear_allocator<int>;
	using vector_type = gpu::vector<int, allocator_type>;
	using size_type = typename vector_type::size_type;

	GPU_SHARED gpu::array<char, sizeof(int) * 512 * 4> local_memory;
	GPU_SHARED allocator_type alloc;

	gpu::block_t block = gpu::this_thread_block();
	if (block.thread_rank() == 0)
		new (&alloc) allocator_type(local_memory.data(), local_memory.size());
	block.sync();

	{
		GPU_SHARED vector_type vec;
		if (block.thread_rank() == 0)
			new (&vec) vector_type(alloc);
		block.sync();

		auto thid = block.thread_rank();
		vec.push_back(block, thid);

		ENSURE(vec.size() == block.size());
		ENSURE(vec.capacity() >= block.size());

		ENSURE(vec.front() == 0);
		ENSURE(vec.back() == block.size() - 1);

		vec.emplace_back(block, thid + 3);
		ENSURE(vec[block.size() + thid] == thid + 3);

		vec.clear(block);

		ENSURE(vec.size() == 0);
		ENSURE(vec.capacity() >= block.size());

		vec.push_back(block, thid + 5);

		ENSURE(vec[thid] == thid + 5);
		ENSURE(vec.size() == block.size());
		ENSURE(vec.capacity() >= block.size());

		vec.push_back(block, thid + 6);
		ENSURE(vec[block.size() + thid] == thid + 6);
		vec.pop_back(block);

		ENSURE(vec[thid] == thid + 5);
	}
}

GPU_GLOBAL void test_vector_constructor_thread()
{
	using allocator_type = gpu::linear_allocator<int>;
	using vector_type = gpu::vector<int, allocator_type>;
	using size_type = typename vector_type::size_type;

	gpu::array<char, sizeof(int) * 10> local_memory;
	allocator_type alloc(local_memory.data(), local_memory.size());

	{
		vector_type vec(alloc);
		ENSURE(vec.capacity() == 0);
		ENSURE(vec.empty());
		ENSURE(vec.size() == 0);
	}

	{
		size_type count = 7;
		vector_type vec(count, 5, alloc);
		ENSURE(vec.capacity() >= count);
		ENSURE(!vec.empty());
		ENSURE(vec.size() == count);
	}
}

GPU_GLOBAL void test_vector_push_back_thread()
{
	using allocator_type = gpu::linear_allocator<int>;
	using vector_type = gpu::vector<int, allocator_type>;
	using size_type = typename vector_type::size_type;

	gpu::array<char, sizeof(int) * 10> local_memory;
	allocator_type alloc(local_memory.data(), local_memory.size());

	auto thid = gpu::this_thread_block().thread_rank();

	{
		vector_type vec(alloc);
		vec.push_back(thid);

		ENSURE(vec.front() == thid);
		ENSURE(vec.back() == thid);

		vec.emplace_back(thid + 1);
		ENSURE(vec.front() == thid);
		ENSURE(vec.back() == thid + 1);

		ENSURE(vec[0] == thid);
		ENSURE(vec[1] == thid + 1);

		ENSURE(vec.size() == 2);
		ENSURE(vec.capacity() >= 2);

		vec.clear();

		ENSURE(vec.size() == 0);
		ENSURE(vec.capacity() >= 2);

		vec.push_back(thid + 2);
		vec.push_back(thid + 3);

		ENSURE(vec.front() == thid + 2);
		ENSURE(vec.back() == thid + 3);

		ENSURE(vec.size() == 2);
		ENSURE(vec.capacity() >= 2);

		vec.pop_back();

		ENSURE(vec.front() == thid + 2);
		ENSURE(vec.size() == 1);
		ENSURE(vec.capacity() >= 2);
	}
}

TEST_CASE("VECTOR", "[VECTOR][CONTAINERS]")
{
	SECTION("Vector per thread")
	{
		CHECK(launch(test_vector_constructor_thread));
		CHECK(launch(test_vector_push_back_thread));
	}

	SECTION("Vector per thread")
	{
		CHECK(launch(test_vector_constructor_block));
		CHECK(launch(test_vector_push_back_block));
	}
}
