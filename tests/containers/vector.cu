#include <gstl/containers/vector.cuh>
#include <Catch2/catch.hpp>

#include <base_test.cuh>

#include <gstl/algorithms/fill.cuh>
#include <gstl/containers/array.cuh>

GPU_GLOBAL void test_vector_constructor_block()
{
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
	}
}

TEST_CASE("VECTOR", "[VECTOR][CONTAINERS]")
{
	SECTION("Vector functions")
	{
		CHECK(launch(test_vector_constructor_thread));
		CHECK(launch(test_vector_push_back_thread));
	}
}
