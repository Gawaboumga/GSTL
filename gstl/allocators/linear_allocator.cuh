#pragma once

#ifndef GPU_ALLOCATORS_LINEAR_ALLOCATOR_HPP
#define GPU_ALLOCATORS_LINEAR_ALLOCATOR_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/allocators/base_allocator.cuh>

namespace gpu
{
	template <typename T>
	class linear_allocator : public base_allocator<linear_allocator<T>>
	{
		public:
			using byte_type = typename base_allocator<linear_allocator<T>>::byte_type;
			using size_type = typename base_allocator<linear_allocator<T>>::size_type;
			using value_type = T;
			using pointer = allocated_memory<T>;

			GPU_DEVICE linear_allocator() = default;
			GPU_DEVICE linear_allocator(const linear_allocator&) = default;
			GPU_DEVICE linear_allocator(byte_type* memory, size_type total_size);

			GPU_DEVICE allocated_memory<T> allocate(block_t g, size_type n);
			template <class BlockTile>
			GPU_DEVICE allocated_memory<T> allocate(BlockTile g, size_type n);
			GPU_DEVICE allocated_memory<T> allocate(size_type n);

			GPU_DEVICE void clear(block_t g);
			template <class BlockTile>
			GPU_DEVICE void clear(BlockTile g);
			GPU_DEVICE void clear();

			GPU_DEVICE void deallocate(block_t g, allocated_memory<T>& ptr, size_type n);
			template <class BlockTile>
			GPU_DEVICE void deallocate(BlockTile g, allocated_memory<T>& ptr, size_type n);
			GPU_DEVICE void deallocate(allocated_memory<T>& ptr, size_type n);

			GPU_DEVICE size_type memory_consumed() const;

			GPU_DEVICE linear_allocator& operator=(const linear_allocator&) = default;

		private:
			size_type m_offset;
	};
}

#include <gstl/allocators/linear_allocator.cu>

#endif // GPU_ALLOCATORS_LINEAR_ALLOCATOR_HPP
