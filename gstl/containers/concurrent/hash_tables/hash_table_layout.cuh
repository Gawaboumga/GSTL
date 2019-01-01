#pragma once

#ifndef GPU_CONTAINERS_CONCURRENT_HASH_TABLES_HASH_TABLE_INFO_HPP
#define GPU_CONTAINERS_CONCURRENT_HASH_TABLES_HASH_TABLE_INFO_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/allocators/allocated_memory.cuh>
#include <gstl/functional/function_object.cuh>
#include <gstl/memory/allocator_traits.cuh>
#include <gstl/utility/atomic.cuh>
#include <gstl/utility/hash.cuh>
#include <gstl/utility/pair.cuh>

namespace gpu
{
	namespace concurrent
	{
		enum Entry
		{
			Entry_Busy,
			Entry_Deleted,
			Entry_Free,
			Entry_Occupied,
		};

		template <typename Key, typename T>
		struct default_layout_type
		{
			atomic<Entry> entry;
			pair<Key, T> data;
		};

		template <typename Key, typename T, class Allocator>
		class hash_table_layout
		{
			public:
				using key_type = Key;
				using mapped_type = T;
				using value_type = pair<key_type, mapped_type>;
				using allocator_type = Allocator;
				using allocated_type = typename allocator_traits<allocator_type>::pointer;
				using size_type = typename allocator_traits<allocator_type>::size_type;
				using difference_type = typename allocator_traits<allocator_type>::difference_type;
				using lock_type = Entry;
				using layout_type = typename allocator_traits<allocator_type>::value_type;

				using hasher = gpu::hash<key_type>;
				using key_equal = gpu::equal_to<key_type>;
				using pointer = value_type*;
				using const_pointer = const value_type*;
				using iterator = pointer;
				using const_iterator = const_pointer;

				template <class Thread>
				GPU_DEVICE allocated_type allocate(Thread g, allocator_type& allocator, size_type count);
				GPU_DEVICE void assign(allocated_type&& pointer);

				GPU_DEVICE bool busy(lock_type lock_info);

				template <class Thread>
				GPU_DEVICE void deallocate(Thread g, allocator_type& allocator, allocated_type ptr);
				GPU_DEVICE static void default_transfer(value_type& lhs, value_type&& rhs);
				GPU_DEVICE static void default_transfer_const(value_type& lhs, const value_type& rhs);
				GPU_DEVICE void destroy(size_type index);

				GPU_DEVICE bool empty(lock_type lock_info);
				template <class Thread>
				GPU_DEVICE bool empty(Thread g, lock_type lock_info);
				GPU_DEVICE bool empty_lock(lock_type lock_info, size_type index, const key_type& key);

				template <class Thread>
				GPU_DEVICE void fill_empty(Thread g, size_type size);

				GPU_DEVICE iterator get_data(size_type index);
				GPU_DEVICE lock_type get_lock_info(size_type index);
				GPU_DEVICE size_type get_position(const key_type& key, size_type offset);
				template <class Thread>
				GPU_DEVICE size_type get_position(Thread g, const key_type& key, size_type offset);
				GPU_DEVICE value_type& get_value(size_type index);

				template <class Thread>
				GPU_DEVICE size_type increment_offset(Thread g, size_type offset);

				GPU_DEVICE bool match(size_type index, const key_type& key);
				template <class Thread>
				GPU_DEVICE bool match(Thread g, size_type index, const key_type& key, size_type* winning_index);
				GPU_DEVICE bool match_lock(lock_type lock_info, size_type index, const key_type& key);

				GPU_DEVICE bool occupied(size_type index, const key_type& key);

				GPU_DEVICE void unlock(size_type index);

			private:
				GPU_DEVICE layout_type& get_lock(size_type index);

				allocated_type m_storage;
		};
	}
}

#include <gstl/containers/concurrent/hash_tables/hash_table_layout.cu>

#endif // GPU_CONTAINERS_CONCURRENT_HASH_TABLES_HASH_TABLE_INFO_HPP
