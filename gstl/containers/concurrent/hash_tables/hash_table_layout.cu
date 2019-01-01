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
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::busy(lock_type lock_info)
		{
			return lock_info == Entry_Busy;
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
		GPU_DEVICE void hash_table_layout<Key, T, Allocator>::default_transfer_const(value_type& lhs, const value_type& rhs)
		{
			lhs = rhs;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE void hash_table_layout<Key, T, Allocator>::destroy(size_type index)
		{
			layout_type& layout = get_lock(index);
			layout.entry.store(Entry_Deleted);
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::empty(lock_type lock_info)
		{
			return lock_info == Entry_Free;
		}

		template <typename Key, typename T, class Allocator>
		template <class Thread>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::empty(Thread g, lock_type lock_info)
		{
			return lock_info == Entry_Free;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::empty_lock(lock_type lock_info, size_type index, const key_type& key)
		{
			if (lock_info != Entry_Free)
				return false;

			layout_type& layout = get_lock(index);
			return layout.entry.compare_and_swap(Entry_Free, Entry_Busy) == Entry_Free;
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
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::iterator hash_table_layout<Key, T, Allocator>::get_data(size_type index)
		{
			layout_type& layout = get_lock(index);
			return &layout.data;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::lock_type hash_table_layout<Key, T, Allocator>::get_lock_info(size_type index)
		{
			layout_type& layout = get_lock(index);
			return layout.entry;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::size_type hash_table_layout<Key, T, Allocator>::get_position(const key_type& key, size_type offset)
		{
			return hasher{}(key) + offset;
		}

		template <typename Key, typename T, class Allocator>
		template <class Thread>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::size_type hash_table_layout<Key, T, Allocator>::get_position(Thread g, const key_type& key, size_type offset)
		{
			return hasher{}(key) + offset + g.thread_rank();
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::value_type& hash_table_layout<Key, T, Allocator>::get_value(size_type index)
		{
			layout_type& layout = get_lock(index);
			return layout.data;
		}

		template <typename Key, typename T, class Allocator>
		template <class Thread>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::size_type hash_table_layout<Key, T, Allocator>::increment_offset(Thread g, size_type offset)
		{
			return offset + g.size();
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::match(size_type index, const key_type& key)
		{
			layout_type& layout = get_lock(index);

			if (layout.entry == Entry_Occupied)
				return key_equal{}(layout.data.first, key);

			return false;
		}

		template <typename Key, typename T, class Allocator>
		template <class Thread>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::match(Thread g, size_type index, const key_type& key, size_type* winning_index)
		{
			layout_type& layout = get_lock(index);

			bool result = false;
			if (layout.entry == Entry_Occupied)
				result = key_equal{}(layout.data.first, key);

			auto winning_thid = first_index(g, result);
			*winning_index = shfl(g, index, winning_thid);
			return winning_thid != g.size();
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::match_lock(lock_type lock_info, size_type index, const key_type& key)
		{
			layout_type& layout = get_lock(index);

			if (layout.entry == Entry_Occupied)
				if (key_equal{}(layout.data.first, key))
					return layout.entry.compare_and_swap(Entry_Occupied, Entry_Busy) == Entry_Occupied;

			return false;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE bool hash_table_layout<Key, T, Allocator>::occupied(size_type index, const key_type& key)
		{
			layout_type& layout = get_lock(index);

			if (layout.entry == Entry_Occupied)
				return key_equal{}(layout.data.first, key);

			return true;
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE void hash_table_layout<Key, T, Allocator>::unlock(size_type index)
		{
			layout_type& layout = get_lock(index);
			layout.entry.store(Entry_Occupied);
		}

		template <typename Key, typename T, class Allocator>
		GPU_DEVICE typename hash_table_layout<Key, T, Allocator>::layout_type& hash_table_layout<Key, T, Allocator>::get_lock(size_type offset)
		{
			return m_storage[offset];
		}
	}
}
