#include <gstl/containers/concurrent/hash_tables/hash_table_layout.cuh>

#include <gstl/algorithms/for_each.cuh>

namespace gpu
{
	namespace concurrent
	{
		template <typename Key, typename T, class Allocator>
		template <class Thread>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::allocated_type hash_table_layout<Key, T, Allocator>::allocate(Thread g, allocator_type& allocator, size_type count)
		{
			return allocator_traits<allocator_type>::allocate(g, allocator, count);
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE void hash_table_layout<Key, T, Allocator>::assign(allocated_type&& pointer)
		{
			m_storage = std::move(pointer);
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::busy(const lock_type* lock_info)
		{
			return lock_info->entry == Entry_Busy;
		}

		template <typename Key, typename T, class Allocator>
		template <class Thread>
		GPU_DEVICE void hash_table_layout<Key, T, Allocator>::deallocate(Thread g, allocator_type& allocator, allocated_type ptr)
		{
			return allocator_traits<allocator_type>::deallocate(g, allocator, ptr);
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE void hash_table_layout<Key, T, Allocator>::default_transfer(value_type& lhs, value_type&& rhs)
		{
			lhs = std::move(rhs);
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::empty_lock(lock_type* lock_info, const key_type& key)
		{
			return lock_info->entry.compare_and_swap(Entry_Free, Entry_Busy) == Entry_Free;
		}

		template <typename Key, typename T, class Allocator>
		template <class Thread>
		GPU_DEVICE void hash_table_layout<Key, T, Allocator>::fill_empty(Thread g, size_type size)
		{
			for_each_n(g, m_storage, size, [](default_layout_type<key_type, mapped_type>& layout) {
				layout.entry.store_unatomically(Entry_Free);
			});
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::iterator hash_table_layout<Key, T, Allocator>::get_data(lock_type* lock_info)
		{
			return &lock_info->data;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::lock_type* hash_table_layout<Key, T, Allocator>::get_lock(size_type offset)
		{
			return &m_storage[offset];
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::size_type hash_table_layout<Key, T, Allocator>::get_position(const key_type& key, size_type offset)
		{
			return hasher{}(key) + offset;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::value_type& hash_table_layout<Key, T, Allocator>::get_value(lock_type* lock_info)
		{
			return lock_info->data;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::match_lock(lock_type* lock_info, const key_type& key)
		{
			return lock_info->entry.compare_and_swap(Entry_Occupied, Entry_Busy) == Entry_Occupied;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE void hash_table_layout<Key, T, Allocator>::unlock(lock_type* lock_info)
		{
			lock_info->entry.store(Entry_Occupied);
		}
	}
}
