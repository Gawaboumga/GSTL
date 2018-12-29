#pragma once

#ifndef GPU_CONTAINERS_CONCURRENT_HASH_TABLES_FAST_INTEGER_HPP
#define GPU_CONTAINERS_CONCURRENT_HASH_TABLES_FAST_INTEGER_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/allocators/default_allocator.cuh>
#include <gstl/allocators/linear_allocator.cuh>
#include <gstl/containers/concurrent/hash_tables/hash_table_layout.cuh>

#include <type_traits>

namespace gpu
{
	namespace concurrent
	{
		template <typename Key, typename T, class Allocator = gpu::linear_allocator<default_layout_type<Key, T>>, class HashTableLayout = hash_table_layout<Key, T, Allocator>>
		class fast_integer : private HashTableLayout
		{
			private:
				using layout_type = HashTableLayout;

			public:
				using key_type = Key;
				using mapped_type = T;
				using value_type = typename layout_type::value_type;
				using lock_type = typename layout_type::lock_type;
				using allocator_type = Allocator;
				using allocated_type = typename layout_type::allocated_type;
				using size_type = typename gpu::allocator_traits<allocator_type>::size_type;
				using difference_type = typename gpu::allocator_traits<allocator_type>::difference_type;
				using hasher = typename layout_type::hasher;
				using key_equal = typename layout_type::key_equal;
				using reference = value_type&;
				using const_reference = const value_type&;
				using pointer = typename layout_type::pointer;
				using const_pointer = typename layout_type::const_pointer;
				using iterator = typename layout_type::iterator;
				using const_iterator = typename layout_type::const_pointer;

			public:
				GPU_DEVICE iterator end();
				GPU_DEVICE const_iterator end() const;
				GPU_DEVICE const_iterator cend() const;

				fast_integer() = default;
				template <class Thread>
				GPU_DEVICE fast_integer(Thread g, allocator_type& allocator, size_type expected_capacity);

				GPU_DEVICE size_type capacity() const;
				GPU_DEVICE bool contains(const key_type& key) const;
				template <class Thread>
				GPU_DEVICE bool contains(Thread g, const key_type& key) const;
				template <class Thread>
				GPU_DEVICE void clear(Thread g);

				GPU_DEVICE bool empty() const;
				GPU_DEVICE void erase(const key_type& key);
				template <class Thread>
				GPU_DEVICE void erase(Thread g, const key_type& key);
				GPU_DEVICE void erase(const_iterator pos);
				template <class Thread>
				GPU_DEVICE void erase(Thread g, const_iterator pos);

				GPU_DEVICE iterator find(const key_type& key);
				template <class Thread>
				GPU_DEVICE iterator find(Thread g, const key_type& key);
				GPU_DEVICE const_iterator find(const key_type& key) const;
				template <class Thread>
				GPU_DEVICE const_iterator find(Thread g, const key_type& key) const;

				GPU_DEVICE iterator insert(const value_type& value);
				template <class Thread>
				GPU_DEVICE iterator insert(Thread g, const value_type& value);
				GPU_DEVICE iterator insert(value_type&& value);
				template <class Thread>
				GPU_DEVICE iterator insert(Thread g, value_type&& value);
				template <class Function>
				GPU_DEVICE iterator insert_or_update(const value_type& value, Function f);
				template <class Thread, class Function>
				GPU_DEVICE iterator insert_or_update(Thread g, const value_type& value, Function f);
				template <class Function>
				GPU_DEVICE iterator insert_or_update(value_type&& value, Function f);
				template <class Thread, class Function>
				GPU_DEVICE iterator insert_or_update(Thread g, value_type&& value, Function f);

				GPU_DEVICE T& operator[](const key_type& key);
				GPU_DEVICE const T& operator[](const key_type& key) const;

				template <class Thread>
				GPU_DEVICE void reserve(Thread g, allocator_type& allocator, size_type count);
				template <class Thread>
				GPU_DEVICE void resize(Thread g, allocator_type& allocator, size_type new_capacity);

				GPU_DEVICE size_type size() const;

				GPU_DEVICE void debug() const;

			private:
				using alloc_traits = allocator_traits<allocator_type>;

				size_type m_mask;
				gpu::atomic<size_type> m_number_of_elements;
		};
	}
}

#include <gstl/containers/concurrent/hash_tables/fixed/fast_integer.cu>

#endif // GPU_CONTAINERS_CONCURRENT_HASH_TABLES_FAST_INTEGER_HPP
