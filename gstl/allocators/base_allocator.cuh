#pragma once

#ifndef GPU_ALLOCATORS_BASE_ALLOCATOR_HPP
#define GPU_ALLOCATORS_BASE_ALLOCATOR_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/allocators/allocated_memory.cuh>
#include <gstl/containers/array.cuh>

namespace gpu
{
	template <class DerivedAllocator>
	class base_allocator
	{
		public:
			using byte_type = char;
			template <typename T>
			using pointer = allocated_memory<T>;
			using size_type = size_t;
			using difference_type = ptrdiff_t;

			static constexpr size_type DEFAULT_PADDING = DEFAULT_BYTE_ALIGNMENT;

		public:
			GPU_DEVICE byte_type* begin() const;
			GPU_DEVICE byte_type* end() const;

			base_allocator() = default;
			base_allocator(const base_allocator&) = default;
			template <unsigned int N>
			GPU_DEVICE base_allocator(array<byte_type, N>& fixed_memory);
			GPU_DEVICE base_allocator(byte_type* memory, size_type total_size);
			base_allocator(base_allocator&& other) = default;

			template <typename T>
			GPU_DEVICE pointer<T> allocate(block_t g, size_type n = 1);
			template <class BlockTile, typename T>
			GPU_DEVICE pointer<T> allocate(BlockTile g, size_type n = 1);
			template <typename T>
			GPU_DEVICE pointer<T> allocate(size_type n = 1);

			GPU_DEVICE void clear(block_t g);
			template <class BlockTile>
			GPU_DEVICE void clear(BlockTile g);
			GPU_DEVICE void clear();

			template <typename T>
			GPU_DEVICE void deallocate(block_t g, pointer<T>& ptr, size_type n = 1);
			template <class BlockTile, typename T>
			GPU_DEVICE void deallocate(BlockTile g, pointer<T>& ptr, size_type n = 1);
			template <typename T>
			GPU_DEVICE void deallocate(pointer<T>& ptr, size_type n = 1);

			base_allocator& operator=(const base_allocator& other) = default;
			base_allocator& operator=(base_allocator&& other) = default;

		protected:
			byte_type* m_memory;
			size_type m_total_size;
	};
}

#include <gstl/allocators/base_allocator.cu>

#endif // GPU_ALLOCATORS_BASE_ALLOCATOR_HPP
