#pragma once

#ifndef GPU_CONTAINERS_VECTOR_HPP
#define GPU_CONTAINERS_VECTOR_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/allocators/default_allocator.cuh>
#include <gstl/memory/allocator_traits.cuh>
#include <gstl/utility/iterator.cuh>

namespace gpu
{
	template <typename T, class Allocator = allocator<T>>
	class vector
	{
		public:
			using value_type = T;
			using allocator_type = Allocator;
			using size_type = typename allocator_traits<allocator_type>::size_type;
			using reference = value_type&;
			using const_reference = const value_type&;
			using difference_type = typename allocator_traits<allocator_type>::difference_type;
			using pointer = typename allocator_traits<allocator_type>::pointer;
			using const_pointer = typename allocator_traits<allocator_type>::const_pointer;
			using iterator = T*;
			using const_iterator = const T*;
			using reverse_iterator = gpu::reverse_iterator<iterator>;
			using const_reverse_iterator = gpu::reverse_iterator<const_iterator>;

		public:
			GPU_DEVICE GPU_CONSTEXPR iterator begin();
			GPU_DEVICE GPU_CONSTEXPR const_iterator begin() const;
			GPU_DEVICE GPU_CONSTEXPR const_iterator cbegin() const;
			GPU_DEVICE GPU_CONSTEXPR iterator end();
			GPU_DEVICE GPU_CONSTEXPR const_iterator end() const;
			GPU_DEVICE GPU_CONSTEXPR const_iterator cend() const;

			GPU_DEVICE GPU_CONSTEXPR reverse_iterator rbegin();
			GPU_DEVICE GPU_CONSTEXPR const_reverse_iterator rbegin() const;
			GPU_DEVICE GPU_CONSTEXPR const_reverse_iterator crbegin() const;
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator rend();
			GPU_DEVICE GPU_CONSTEXPR const_reverse_iterator rend() const;
			GPU_DEVICE GPU_CONSTEXPR const_reverse_iterator crend() const;

		public:
			vector() noexcept = default;
			GPU_DEVICE explicit vector(const Allocator& alloc) noexcept;
			GPU_DEVICE explicit vector(Allocator&& alloc) noexcept;
			GPU_DEVICE vector(size_type count, const T& value, const Allocator& alloc = Allocator());
			template <class Thread>
			GPU_DEVICE vector(Thread g, size_type count, const T& value, const Allocator& alloc = Allocator());
			GPU_DEVICE explicit vector(size_type count, const Allocator& alloc = Allocator());
			template <class Thread>
			GPU_DEVICE vector(Thread g, size_type count, const Allocator& alloc = Allocator());
			template <class ForwardIt>
			GPU_DEVICE vector(ForwardIt first, ForwardIt last, const Allocator& alloc = Allocator());
			template <class Thread, class ForwardIt>
			GPU_DEVICE vector(Thread g, ForwardIt first, ForwardIt last, const Allocator& alloc = Allocator());
			GPU_DEVICE vector(vector&& other) noexcept;
			GPU_DEVICE vector(vector&& other, const Allocator& alloc) noexcept;
			template <class Thread>
			GPU_DEVICE void destruct(Thread g);

			GPU_DEVICE void assign(size_type count, const T& value);
			template <class Thread>
			GPU_DEVICE void assign(Thread g, size_type count, const T& value);
			template <class ForwardIt>
			GPU_DEVICE void assign(ForwardIt first, ForwardIt last);
			template <class Thread, class ForwardIt>
			GPU_DEVICE void assign(Thread g, ForwardIt first, ForwardIt last);

			GPU_DEVICE reference back();
			GPU_DEVICE const_reference back() const;

			GPU_DEVICE size_type capacity() const noexcept;
			GPU_DEVICE void clear();
			template <class Thread>
			GPU_DEVICE void clear(Thread g);

			GPU_DEVICE value_type* data() noexcept;
			GPU_DEVICE const value_type* data() const noexcept;

			template <class... Args>
			GPU_DEVICE reference emplace_back(Args&&... args);
			template <class... Args>
			GPU_DEVICE reference emplace_back(block_t g, Args&&... args);
			template <class... Args, unsigned int tile_sz>
			GPU_DEVICE reference emplace_back(block_tile_t<tile_sz> g, Args&&... args);
			GPU_DEVICE bool empty() const noexcept;

			GPU_DEVICE reference front();
			GPU_DEVICE const_reference front() const;

			GPU_DEVICE allocator_type& get_allocator();
			GPU_DEVICE const allocator_type& get_allocator() const;

			GPU_DEVICE reference operator[](size_type n);
			GPU_DEVICE const_reference operator[](size_type n) const;
			GPU_DEVICE vector& operator=(vector&& other);

			GPU_DEVICE void pop_back();
			GPU_DEVICE bool pop_back(T* result);
			template <class Thread>
			GPU_DEVICE void pop_back(Thread g);
			template <class Thread>
			GPU_DEVICE bool pop_back(Thread g, T* result);
			GPU_DEVICE void push_back(const_reference value);
			template <class Thread>
			GPU_DEVICE void push_back(Thread g, const_reference value);
			GPU_DEVICE void push_back(value_type&& value);
			template <class Thread>
			GPU_DEVICE void push_back(Thread g, value_type&& value);

			GPU_DEVICE void reserve(size_type n);
			template <class Thread>
			GPU_DEVICE void reserve(Thread g, size_type n);

			GPU_DEVICE void shrink_to_fit();
			template <class Thread>
			GPU_DEVICE void shrink_to_fit(Thread g);
			GPU_DEVICE size_type size() const noexcept;
			GPU_DEVICE void swap(vector& other);

		private:
			using alloc_traits = allocator_traits<allocator_type>;

			GPU_DEVICE pointer allocate(size_type n);
			template <class Thread>
			GPU_DEVICE pointer allocate(Thread g, size_type n);

			GPU_DEVICE void deallocate(pointer ptr);
			template <class Thread>
			GPU_DEVICE void deallocate(Thread g, pointer ptr);

			template <typename... Args>
			GPU_DEVICE void insert_value_end(Args&&... args);
			template <typename... Args>
			GPU_DEVICE void insert_value_end(block_t g, Args&&... args);
			template <typename... Args, unsigned int tile_sz>
			GPU_DEVICE void insert_value_end(block_tile_t<tile_sz> g, Args&&... args);

			pointer m_begin;
			pointer m_end;
			pointer m_end_capacity;
			allocator_type m_allocator;
	};
}

#include <gstl/containers/vector.cu>

#endif // GPU_CONTAINERS_VECTOR_HPP
