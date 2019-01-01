#include <gstl/containers/concurrent/hash_tables/fixed/fast_integer.cuh>

#include <gstl/math/bit.cuh>

namespace gpu
{
	namespace concurrent
	{
		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::end()
		{
			return iterator(nullptr);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::const_iterator fast_integer<Key, T, Allocator, HashInfo>::end() const
		{
			return const_iterator(nullptr);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::const_iterator fast_integer<Key, T, Allocator, HashInfo>::cend() const
		{
			return const_iterator(nullptr);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE fast_integer<Key, T, Allocator, HashInfo>::fast_integer(Thread g, allocator_type& allocator, size_type expected_capacity)
		{
			size_type power_of_two = ceil2(expected_capacity);
			allocated_type ptr = layout_type::allocate(g, allocator, power_of_two);
			if (g.thread_rank() == 0)
			{
				m_number_of_elements.store_unatomically(0);
				m_mask = power_of_two - 1u;
				layout_type::assign(std::move(ptr));
			}
			g.sync();
			layout_type::fill_empty(g, capacity());
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::size_type fast_integer<Key, T, Allocator, HashInfo>::capacity() const
		{
			return size_type(m_mask + 1);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE bool fast_integer<Key, T, Allocator, HashInfo>::contains(const key_type& key) const
		{
			return find(key) != end();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE bool fast_integer<Key, T, Allocator, HashInfo>::contains(Thread g, const key_type& key) const
		{
			return find(g, key) != end();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE void fast_integer<Key, T, Allocator, HashInfo>::clear(Thread g)
		{
			layout_type::fill_empty(g, capacity());
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE bool fast_integer<Key, T, Allocator, HashInfo>::empty() const
		{
			return size() == 0;
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE void fast_integer<Key, T, Allocator, HashInfo>::erase(const key_type& key)
		{
			size_type offset = 0;
			do
			{
				size_type index = layout_type::get_position(key, offset) & m_mask;
				lock_type current_lock;
				do
				{
					current_lock = layout_type::get_lock_info(index);

					if (layout_type::match_lock(current_lock, index, key))
					{
						layout_type::destroy(index);
						--m_number_of_elements;
						return;
					}
					if (layout_type::empty(current_lock))
						return;
				} while (layout_type::busy(current_lock));

				++offset;
			} while (offset < capacity());

		#if defined(GPU_DEBUG_FAST_INTEGER) || defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(false, "Could not find element");
		#endif // GPU_DEBUG_FAST_INTEGER || GPU_DEBUG_OUT_OF_RANGE
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE void fast_integer<Key, T, Allocator, HashInfo>::erase(Thread g, const key_type& key)
		{
			if (g.thread_rank() == 0)
				erase(key);
			g.sync();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE void fast_integer<Key, T, Allocator, HashInfo>::erase(const_iterator pos)
		{

		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE void fast_integer<Key, T, Allocator, HashInfo>::erase(Thread g, const_iterator pos)
		{

		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::find(const key_type& key)
		{
			size_type offset = 0;
			do
			{
				size_type index = layout_type::get_position(key, offset) & m_mask;
				lock_type current_lock;
				do
				{
					current_lock = layout_type::get_lock_info(index);

					if (layout_type::match(index, key))
						return layout_type::get_data(index);
					if (layout_type::empty(current_lock))
						return end();
				} while (layout_type::busy(current_lock));

				++offset;
			} while (offset < capacity());

		#if defined(GPU_DEBUG_FAST_INTEGER) || defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(false, "Could not find element");
		#endif // GPU_DEBUG_FAST_INTEGER || GPU_DEBUG_OUT_OF_RANGE
			return end();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::find(Thread g, const key_type& key)
		{
			size_type offset = 0;
			do
			{
				size_type index = layout_type::get_position(g, key, offset) & m_mask;
				lock_type current_lock;
				do
				{
					current_lock = layout_type::get_lock_info(index);

					size_type winning_index;
					if (layout_type::match(g, index, key, &winning_index))
						return layout_type::get_data(winning_index);
					if (layout_type::empty(g, current_lock))
						return end();
				} while (layout_type::busy(current_lock));

				offset = layout_type::increment_offset(g, offset);
			} while (offset < capacity());

		#if defined(GPU_DEBUG_FAST_INTEGER) || defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(false, "Could not find element");
		#endif // GPU_DEBUG_FAST_INTEGER || GPU_DEBUG_OUT_OF_RANGE
			return end();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::const_iterator fast_integer<Key, T, Allocator, HashInfo>::find(const key_type& key) const
		{
			size_type offset = 0;
			do
			{
				size_type index = layout_type::get_position(key, offset) & m_mask;
				lock_type current_lock;
				do
				{
					current_lock = layout_type::get_lock_info(index);

					if (layout_type::match(index, key))
						return layout_type::get_data(index);
					if (layout_type::empty(current_lock))
						return end();
				} while (layout_type::busy(current_lock));

				++offset;
			} while (offset < capacity());

		#if defined(GPU_DEBUG_FAST_INTEGER) || defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(false, "Could not find element");
		#endif // GPU_DEBUG_FAST_INTEGER || GPU_DEBUG_OUT_OF_RANGE
			return end();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::const_iterator fast_integer<Key, T, Allocator, HashInfo>::find(Thread g, const key_type& key) const
		{
			size_type offset = 0;
			do
			{
				size_type index = layout_type::get_position(g, key, offset) & m_mask;
				lock_type* current_lock;
				do
				{
					current_lock = layout_type::get_lock(g, index);
					size_type new_index;
					if (layout_type::match_lock(g, current_lock, key, &new_index))
						return layout_type::get_data(g, new_index);
					if (layout_type::empty(g, current_lock))
						return end();
				} while (layout_type::busy(current_lock));

				offset = layout_type::increment_offset(g, offset);
			} while (offset < capacity());

		#if defined(GPU_DEBUG_FAST_INTEGER) || defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(false, "Could not find element");
		#endif // GPU_DEBUG_FAST_INTEGER || GPU_DEBUG_OUT_OF_RANGE
			return end();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::insert(const value_type& value)
		{
			return insert_or_update(value, layout_type::default_transfer_const);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::insert(Thread g, const value_type& value)
		{
			return insert_or_update(g, value, layout_type::default_transfer_const);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::insert(value_type&& value)
		{
			return insert_or_update(std::move(value), layout_type::default_transfer);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::insert(Thread g, value_type&& value)
		{
			return insert_or_update(g, std::move(value), layout_type::default_transfer);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Function>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::insert_or_update(const value_type& value, Function f)
		{
			size_type offset = 0;
			do
			{
				size_type index = layout_type::get_position(value.first, offset) & m_mask;
				lock_type current_lock;
				bool result = false;
				do
				{
					current_lock = layout_type::get_lock_info(index);

					if (layout_type::empty_lock(current_lock, index, value.first))
					{
						f(layout_type::get_value(index), value);
						layout_type::unlock(index);
						++m_number_of_elements;
						result = true;
					}

					if (!result && layout_type::match_lock(current_lock, index, value.first))
					{
						f(layout_type::get_value(index), value);
						layout_type::unlock(index);
						result = true;
					}

					if (result)
						return layout_type::get_data(index);
				} while (layout_type::empty(current_lock) || layout_type::busy(current_lock) || layout_type::occupied(index, value.first));

				++offset;
			} while (offset < capacity());

		#if defined(GPU_DEBUG_FAST_INTEGER) || defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(false, "Could not find element");
		#endif // GPU_DEBUG_FAST_INTEGER || GPU_DEBUG_OUT_OF_RANGE
			return end();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread, class Function>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::insert_or_update(Thread g, const value_type& value, Function f)
		{
			iterator result;
			if (g.thread_rank() == 0)
				result = insert_or_update(value, f);
			return shfl(g, result);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Function>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::insert_or_update(value_type&& value, Function f)
		{
			size_type offset = 0;
			do
			{
				size_type index = layout_type::get_position(value.first, offset) & m_mask;
				lock_type current_lock;
				bool result = false;
				do
				{
					current_lock = layout_type::get_lock_info(index);

					if (layout_type::empty_lock(current_lock, index, value.first))
					{
						f(layout_type::get_value(index), std::move(value));
						layout_type::unlock(index);
						++m_number_of_elements;
						result = true;
					}

					if (!result && layout_type::match_lock(current_lock, index, value.first))
					{
						f(layout_type::get_value(index), std::move(value));
						layout_type::unlock(index);
						result = true;
					}

					if (result)
						return layout_type::get_data(index);
				} while (layout_type::empty(current_lock) || layout_type::busy(current_lock) || layout_type::occupied(index, value.first));

				++offset;
			} while (offset < capacity());

		#if defined(GPU_DEBUG_FAST_INTEGER) || defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(false, "Could not find element");
		#endif // GPU_DEBUG_FAST_INTEGER || GPU_DEBUG_OUT_OF_RANGE
			return end();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread, class Function>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::iterator fast_integer<Key, T, Allocator, HashInfo>::insert_or_update(Thread g, value_type&& value, Function f)
		{
			iterator result;
			if (g.thread_rank() == 0)
				result = insert_or_update(std::move(value), f);
			return shfl(g, result);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE T& fast_integer<Key, T, Allocator, HashInfo>::operator[](const key_type& key)
		{
			return layout_type::get_mapped_type(find(key));
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE const T& fast_integer<Key, T, Allocator, HashInfo>::operator[](const key_type& key) const
		{
			return layout_type::get_mapped_type(find(key));
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE void fast_integer<Key, T, Allocator, HashInfo>::reserve(Thread g, allocator_type& allocator, size_type count)
		{
			size_type power_of_two = ceil2(count);
			allocated_type ptr = layout_type::allocate(g, allocator, power_of_two);

			layout_type::fill_empty(g, ptr, ptr + power_of_two);
			layout_type::deallocate(g, allocator);

			if (g.thread_rank() == 0)
			{
				m_number_of_elements.store_unatomically(0);
				m_mask = power_of_two - 1u;
				layout_type::assign(std::move(ptr));
			}
			g.sync();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		template <class Thread>
		GPU_DEVICE void fast_integer<Key, T, Allocator, HashInfo>::resize(Thread g, allocator_type& allocator, size_type new_capacity)
		{
		#if defined(GPU_DEBUG_FAST_INTEGER)
			ENSURE(new_capacity > size());
		#endif // GPU_DEBUG_FAST_INTEGER

			size_type power_of_two = ceil2(new_capacity);
			allocated_type ptr = layout_type::allocate(g, allocator, power_of_two);

			layout_type::fill_empty(g, ptr, ptr + power_of_two);
			layout_type::deallocate(g, allocator);

			if (g.thread_rank() == 0)
			{
				m_number_of_elements.store_unatomically(0);
				m_mask = power_of_two - 1u;
				layout_type::assign(std::move(ptr));
			}
			g.sync();
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE typename fast_integer<Key, T, Allocator, HashInfo>::size_type fast_integer<Key, T, Allocator, HashInfo>::size() const
		{
			return size_type(m_number_of_elements);
		}

		template <typename Key, typename T, class Allocator, class HashInfo>
		GPU_DEVICE void fast_integer<Key, T, Allocator, HashInfo>::debug() const
		{

		}
	}
}
