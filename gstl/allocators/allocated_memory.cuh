#pragma once

#ifndef GPU_ALLOCATORS_ALLOCATED_MEMORY_HPP
#define GPU_ALLOCATORS_ALLOCATED_MEMORY_HPP

#include <gstl/prerequisites.hpp>
#include <gstl/utility/iterator.cuh>

namespace gpu
{
	template <typename T>
	class allocated_memory
	{
		public:
			using value_type = T;
			using size_type = size_t;
			using difference_type = ptrdiff_t;
			using reference = value_type&;
			using const_reference = const value_type&;
			using pointer = value_type*;
			using const_pointer = const value_type*;
			using iterator = value_type*;
			using const_iterator = const value_type*;
			using reverse_iterator = gpu::reverse_iterator<iterator>;
			using const_reverse_iterator = gpu::reverse_iterator<const_iterator>; // "gpu::" To avoid clash name with previous using
			using iterator_category = std::random_access_iterator_tag;

		public:
			GPU_DEVICE iterator begin() noexcept;
			GPU_DEVICE const_iterator begin() const noexcept;
			GPU_DEVICE const_iterator cbegin() const noexcept;
			GPU_DEVICE iterator end() noexcept;
			GPU_DEVICE const_iterator end() const noexcept;
			GPU_DEVICE const_iterator cend() const noexcept;

			GPU_DEVICE reverse_iterator rbegin() noexcept;
			GPU_DEVICE const_reverse_iterator rbegin() const noexcept;
			GPU_DEVICE const_reverse_iterator crbegin() const noexcept;
			GPU_DEVICE reverse_iterator rend() noexcept;
			GPU_DEVICE const_reverse_iterator rend() const noexcept;
			GPU_DEVICE const_reverse_iterator crend() const noexcept;

		public:
			GPU_DEVICE allocated_memory() noexcept = default;
			GPU_DEVICE allocated_memory(const allocated_memory&) noexcept = default;
			GPU_DEVICE allocated_memory(std::nullptr_t) noexcept;
			GPU_DEVICE allocated_memory(block_t g, T* ptr, size_type count) noexcept;
			template <class BlockTile>
			GPU_DEVICE allocated_memory(BlockTile g, T* ptr, size_type count) noexcept;
			GPU_DEVICE allocated_memory(T* ptr, size_type count) noexcept;
			GPU_DEVICE allocated_memory(allocated_memory&& other) noexcept = default;

			GPU_DEVICE T* data() noexcept;
			GPU_DEVICE const T* data() const noexcept;

			GPU_DEVICE bool is_valid() const noexcept;

			GPU_DEVICE allocated_memory& operator=(const allocated_memory& other) noexcept = default;
			GPU_DEVICE allocated_memory& operator=(allocated_memory&& other) noexcept = default;
			GPU_DEVICE reference operator[](size_type pos) noexcept;
			GPU_DEVICE const_reference operator[](size_type pos) const noexcept;
			GPU_DEVICE allocated_memory operator+(size_type pos) const noexcept;
			GPU_DEVICE allocated_memory& operator+=(size_type pos) noexcept;
			GPU_DEVICE allocated_memory& operator++() noexcept;
			GPU_DEVICE allocated_memory operator++(int) noexcept;
			GPU_DEVICE allocated_memory operator-(size_type pos) const noexcept;
			GPU_DEVICE allocated_memory& operator-=(size_type pos) noexcept;
			GPU_DEVICE allocated_memory& operator--() noexcept;
			GPU_DEVICE allocated_memory operator--(int) noexcept;
			GPU_DEVICE bool operator==(const allocated_memory& rhs) const noexcept;
			GPU_DEVICE bool operator!=(const allocated_memory& rhs) const noexcept;
			GPU_DEVICE T& operator*() noexcept;
			GPU_DEVICE const T& operator*() const noexcept;
			GPU_DEVICE T* operator->() noexcept;
			GPU_DEVICE const T* operator->() const noexcept;
			GPU_DEVICE bool operator<(const allocated_memory& rhs) const noexcept;

			GPU_DEVICE void debug() const;
			GPU_DEVICE void invalidate() noexcept;

			template <typename U>
			GPU_DEVICE GPU_CONSTEXPR friend typename allocated_memory<U>::difference_type distance(const allocated_memory<U>& first, const allocated_memory<U>& last);

			template <typename U>
			GPU_DEVICE GPU_CONSTEXPR friend U* to_pointer(allocated_memory<U>& memory);
			template <typename U>
			GPU_DEVICE GPU_CONSTEXPR friend const U* to_pointer(const allocated_memory<U>& memory);

		private:
			GPU_DEVICE allocated_memory(T* start, T* end) noexcept;

			GPU_DEVICE void post_condition() const noexcept;

		#ifdef GPU_DEBUG_ALLOCATED_MEMORY
			T* m_start;
			T* m_end;
		#else
			T* m_ptr;
		#endif // GPU_DEBUG_ALLOCATED_MEMORY
	};
}

#include <gstl/allocators/allocated_memory.cu>

#endif // GPU_ALLOCATORS_ALLOCATED_MEMORY_HPP
