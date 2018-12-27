#include <gstl/containers/vector.cuh>

#include <gstl/memory/algorithms.cuh>

namespace gpu
{
	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::iterator vector<T, Allocator>::begin()
	{
		return iterator(data());
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::const_iterator vector<T, Allocator>::begin() const
	{
		return const_iterator(data());
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::const_iterator vector<T, Allocator>::cbegin() const
	{
		return begin();
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::iterator vector<T, Allocator>::end()
	{
		return iterator(data() + N);
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::const_iterator vector<T, Allocator>::end() const
	{
		return const_iterator(data() + N);
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::const_iterator vector<T, Allocator>::cend() const
	{
		return end();
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::reverse_iterator vector<T, Allocator>::rbegin()
	{
		return reverse_iterator(end());
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::const_reverse_iterator vector<T, Allocator>::rbegin() const
	{
		return const_reverse_iterator(end());
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::const_reverse_iterator vector<T, Allocator>::crbegin() const
	{
		return rbegin();
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::reverse_iterator vector<T, Allocator>::rend()
	{
		return reverse_iterator(begin());
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::const_reverse_iterator vector<T, Allocator>::rend() const
	{
		return const_reverse_iterator(begin());
	}

	template <typename T, class Allocator>
	GPU_DEVICE GPU_CONSTEXPR typename vector<T, Allocator>::const_reverse_iterator vector<T, Allocator>::crend() const
	{
		return rend();
	}

	template <typename T, class Allocator>
	GPU_DEVICE vector<T, Allocator>::vector(const allocator_type& allocator) noexcept :
		m_begin{ nullptr },
		m_end{ nullptr },
		m_end_capacity{ nullptr },
		m_allocator{ allocator }
	{
	}

	template <typename T, class Allocator>
	GPU_DEVICE vector<T, Allocator>::vector(allocator_type&& allocator) noexcept :
		m_begin{ nullptr },
		m_end{ nullptr },
		m_end_capacity{ nullptr },
		m_allocator{ std::move(allocator) }
	{
	}

	template <typename T, class Allocator>
	GPU_DEVICE vector<T, Allocator>::vector(size_type count, const T& value, const Allocator& alloc) :
		vector(alloc)
	{
		if (count > 0)
		{
			m_begin = allocate(count);
			m_end = m_begin;
			m_end_capacity = m_begin + count;

			uninitialized_fill_n(m_begin, count, value);
			m_end = m_begin + count;
		}
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE vector<T, Allocator>::vector(Thread g, size_type count, const T& value, const Allocator& alloc)
	{
		if (g.thread_rank() == 0)
			m_allocator = alloc;

		if (count > 0)
		{
			pointer ptr = allocate(g, count);
			if (g.thread_rank() == 0)
			{
				m_begin = ptr;
				m_end = ptr;
				m_end_capacity = m_begin + count;
			}
			g.sync();

			uninitialized_fill_n(g, m_begin, count, value);
			if (g.thread_rank() == 0)
				m_end = m_begin + count;
			g.sync();
		}
	}

	template <typename T, class Allocator>
	GPU_DEVICE vector<T, Allocator>::vector(size_type count, const Allocator& alloc) :
		vector(alloc)
	{
		if (count > 0)
		{
			m_begin = allocate(n);
			m_end = m_begin;
			m_end_capacity = m_begin + n;

			uninitialized_default_construct_n(m_begin, count);
			m_end = m_begin + count;
		}
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE vector<T, Allocator>::vector(Thread g, size_type count, const Allocator& alloc)
	{
		if (g.thread_rank() == 0)
			m_allocator = alloc;

		if (count > 0)
		{
			pointer ptr = allocate(g, count);
			if (g.thread_rank() == 0)
			{
				m_begin = ptr;
				m_end = ptr;
				m_end_capacity = m_begin + count;
			}
			g.sync();

			uninitialized_default_construct_n(g, m_begin, count);
			m_end = m_begin + count;
		}
	}

	template <typename T, class Allocator>
	template <class ForwardIt>
	GPU_DEVICE vector<T, Allocator>::vector(ForwardIt first, ForwardIt last, const Allocator& alloc) :
		vector(alloc)
	{
		auto len = distance(first, last);
		if (len > 0)
		{
			m_begin = allocate(g, len);
			m_end = m_begin;
			m_end_capacity = m_begin + len;

			uninitialized_copy(first, last, m_begin);
			m_end = m_begin + len;
		}
	}

	template <typename T, class Allocator>
	template <class Thread, class ForwardIt>
	GPU_DEVICE vector<T, Allocator>::vector(Thread g, ForwardIt first, ForwardIt last, const Allocator& alloc)
	{
		if (g.thread_rank() == 0)
			m_allocator = alloc;

		auto len = distance(first, last);
		if (len > 0)
		{
			pointer ptr = allocate(g, len);
			if (g.thread_rank() == 0)
			{
				m_begin = ptr;
				m_end = ptr;
				m_end_capacity = m_begin + len;
			}
			g.sync();

			uninitialized_copy(g, first, last, m_begin);
			m_end = m_begin + len;
		}
	}

	template <typename T, class Allocator>
	GPU_DEVICE vector<T, Allocator>::vector(vector&& other) noexcept
	{
		swap(other);
	}

	template <typename T, class Allocator>
	GPU_DEVICE vector<T, Allocator>::vector(vector&& other, const Allocator& alloc) noexcept :
		vector(alloc)
	{
		if (alloc = other.m_alloc)
		{
			swap(other);
		}
		else
		{
			vector<T, Allocator> temp(std::move(*this));
			temp.swap(x);
		}
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::destruct(Thread g)
	{
		if (m_begin)
		{
			clear(g);
			alloc_traits::deallocate(g, get_allocator(), m_begin, capacity());
		}
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::assign(size_type count, const T& value)
	{
		if (count <= capacity())
		{
			size_type s = size();
			fill_n(m_begin, min(count, s), value);
			if (count > s)
				uninitialized_fill_n(m_begin + s, count - s, value);
			else
				destroy_n(m_begin + count, s - count);
		}
		else
		{
			pointer ptr = allocate(count);
			uninitialized_fill_n(ptr, count, value);
			clear();

			deallocate(m_begin);
			m_begin = ptr;
			m_end = ptr + count;
			m_end_capacity = m_end;
		}
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::assign(Thread g, size_type count, const T& value)
	{
		if (count <= capacity())
		{
			size_type s = size();
			fill_n(g, m_begin, min(count, s), value);
			if (count > s)
				uninitialized_fill_n(g, m_begin + s, count - s, value);
			else
				destroy_n(g, m_begin + count, s - count);
		}
		else
		{
			pointer ptr = allocate(g, count);
			uninitialized_fill_n(g, ptr, count, value);
			clear(g);

			deallocate(g, m_begin);
			if (g.thread_rank() == 0)
			{
				m_begin = ptr;
				m_end = ptr + count;
				m_end_capacity = m_end;
			}
			g.sync();
		}
	}

	template <typename T, class Allocator>
	template <class ForwardIt>
	GPU_DEVICE void vector<T, Allocator>::assign(ForwardIt first, ForwardIt last)
	{
		auto count = distance(first, last);
		if (count <= capacity())
		{
			size_type s = size();
			fill_n(m_begin, min(count, s), value);
			if (count > s)
				uninitialized_fill_n(m_begin + s, count - s, value);
			else
				destroy_n(m_begin + count, s - count);
		}
		else
		{
			pointer ptr = allocate(count);
			uninitialized_fill_n(ptr, count, value);
			clear();

			deallocate(m_begin);
			m_begin = ptr;
			m_end = ptr + count;
			m_end_capacity = m_end;
		}
	}

	template <typename T, class Allocator>
	template <class Thread, class ForwardIt>
	GPU_DEVICE void vector<T, Allocator>::assign(Thread g, ForwardIt first, ForwardIt last)
	{
		auto count = distance(first, last);
		if (count <= capacity())
		{
			size_type s = size();
			fill_n(g, m_begin, min(count, s), value);
			if (count > s)
				uninitialized_fill_n(g, m_begin + s, count - s, value);
			else
				destroy_n(g, m_begin + count, s - count);
		}
		else
		{
			pointer ptr = allocate(g, count);
			uninitialized_fill_n(g, ptr, count, value);
			clear(g);

			deallocate(g, m_begin);
			if (g.thread_rank() == 0)
			{
				m_begin = ptr;
				m_end = ptr + count;
				m_end_capacity = m_end;
			}
			g.sync();
		}
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::reference vector<T, Allocator>::back()
	{
	#if defined(GPU_DEBUG_OUT_OF_RANGE) || defined(GPU_DEBUG_VECTOR)
		ENSURE(!empty());
	#endif // GPU_DEBUG_OUT_OF_RANGE || GPU_DEBUG_VECTOR
		return *(m_end - 1);
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::const_reference vector<T, Allocator>::back() const
	{
	#if defined(GPU_DEBUG_OUT_OF_RANGE) || defined(GPU_DEBUG_VECTOR)
		ENSURE(!empty());
	#endif // GPU_DEBUG_OUT_OF_RANGE || GPU_DEBUG_VECTOR
		return *(m_end - 1);
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::size_type vector<T, Allocator>::capacity() const noexcept
	{
		return static_cast<size_type>(distance(m_begin, m_end_capacity));
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::clear()
	{
		alloc_traits::destroy(get_allocator(), m_begin, m_end);
		m_end = m_begin;
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::clear(Thread g)
	{
		alloc_traits::destroy(g, get_allocator(), m_begin, m_end);
		if (g.thread_rank() == 0)
			m_end = m_begin;
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::value_type* vector<T, Allocator>::data() noexcept
	{
		return m_begin;
	}

	template <typename T, class Allocator>
	GPU_DEVICE const typename vector<T, Allocator>::value_type* vector<T, Allocator>::data() const noexcept
	{
		return m_begin;
	}

	template <typename T, class Allocator>
	template <class... Args>
	GPU_DEVICE typename vector<T, Allocator>::reference vector<T, Allocator>::emplace_back(Args&&... args)
	{
		if (m_end != m_end_capacity)
		{
			alloc_traits::construct(get_allocator(), to_pointer(m_end), std::forward<Args>(args)...);
			++m_end;
		}
		else
			insert_value_end(std::forward<Args>(args)...);

		return back();
	}

	template <typename T, class Allocator>
	template <class... Args>
	GPU_DEVICE typename vector<T, Allocator>::reference vector<T, Allocator>::emplace_back(block_t g, Args&&... args)
	{
		if (size() + g.size() < capacity())
		{
			alloc_traits::construct(get_allocator(), to_pointer(m_end) + g.thread_rank(), std::forward<Args>(args)...);
			if (g.thread_rank() == 0)
				m_end += g.size();
		}
		else
			insert_value_end(g, std::forward<Args>(args)...);

		return *(to_pointer(m_end) - g.size() + g.thread_rank());
	}

	template <typename T, class Allocator>
	template <class... Args, unsigned int tile_sz>
	GPU_DEVICE typename vector<T, Allocator>::reference vector<T, Allocator>::emplace_back(block_tile_t<tile_sz> g, Args&&... args)
	{
		if (size() + g.size() < capacity())
		{
			alloc_traits::construct(get_allocator(), to_pointer(m_end) + g.thread_rank(), std::forward<Args>(args)...);
			advance(g, m_end);
		}
		else
			insert_value_end(g, std::forward<Args>(args)...);

		return back();
	}

	template <typename T, class Allocator>
	GPU_DEVICE bool vector<T, Allocator>::empty() const noexcept
	{
		return m_begin == m_end;
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::reference vector<T, Allocator>::front()
	{
	#if defined(GPU_DEBUG_OUT_OF_RANGE) || defined(GPU_DEBUG_VECTOR)
		ENSURE(!empty());
	#endif // GPU_DEBUG_OUT_OF_RANGE || GPU_DEBUG_VECTOR
		return *m_begin;
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::const_reference vector<T, Allocator>::front() const
	{
	#if defined(GPU_DEBUG_OUT_OF_RANGE) || defined(GPU_DEBUG_VECTOR)
		ENSURE(!empty());
	#endif // GPU_DEBUG_OUT_OF_RANGE || GPU_DEBUG_VECTOR
		return *m_begin;
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::allocator_type& vector<T, Allocator>::get_allocator()
	{
		return m_allocator;
	}

	template <typename T, class Allocator>
	GPU_DEVICE const typename vector<T, Allocator>::allocator_type& vector<T, Allocator>::get_allocator() const
	{
		return m_allocator;
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::reference vector<T, Allocator>::operator[](size_type n)
	{
	#if defined(GPU_DEBUG_OUT_OF_RANGE) || defined(GPU_DEBUG_VECTOR)
		ENSURE(n < size());
	#endif // GPU_DEBUG_OUT_OF_RANGE || GPU_DEBUG_VECTOR
		return m_begin[n];
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::const_reference vector<T, Allocator>::operator[](size_type n) const
	{
	#if defined(GPU_DEBUG_OUT_OF_RANGE) || defined(GPU_DEBUG_VECTOR)
		ENSURE(n < size());
	#endif // GPU_DEBUG_OUT_OF_RANGE || GPU_DEBUG_VECTOR
		return m_begin[n];
	}

	template <typename T, class Allocator>
	GPU_DEVICE vector<T, Allocator>& vector<T, Allocator>::operator=(vector&& other)
	{
		if (this != &other)
			swap(other);

		return *this;
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::pop_back()
	{
	#if defined(GPU_DEBUG_OUT_OF_RANGE) || defined(GPU_DEBUG_VECTOR)
		ENSURE(!empty());
	#endif // GPU_DEBUG_OUT_OF_RANGE || GPU_DEBUG_VECTOR

		--m_end;
		alloc_traits::destroy(get_allocator(), to_pointer(m_end));
	}

	template <typename T, class Allocator>
	GPU_DEVICE bool vector<T, Allocator>::pop_back(T* result)
	{
		if (empty())
			return false;

		--m_end;
		*result = std::move(*m_end);
		return true;
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::pop_back(Thread g)
	{
	#if defined(GPU_DEBUG_OUT_OF_RANGE) || defined(GPU_DEBUG_VECTOR)
		ENSURE(size() >= g.size());
	#endif // GPU_DEBUG_OUT_OF_RANGE || GPU_DEBUG_VECTOR

		if (g.thread_rank() == 0)
			m_end -= g.size();
		g.sync();
		alloc_traits::destroy(get_allocator(), to_pointer(m_end) + g.thread_rank());
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::pop_back(Thread g, T* result)
	{
		if (g.size() >= size())
		{
			if (g.thread_rank() == 0)
				m_end = m_begin;
			g.sync();
			*result = std::move(*(to_pointer(m_end) + g.thread_rank()));
			return g.thread_rank() < size();
		}
		else
		{
			if (g.thread_rank() == 0)
				m_end -= g.size();
			g.sync();
			*result = std::move(*(to_pointer(m_end) + g.thread_rank()));
			return true;
		}
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::push_back(const_reference value)
	{
		if (m_end < m_end_capacity)
		{
			alloc_traits::construct(get_allocator(), to_pointer(m_end), value);
			++m_end;
		}
		else
			insert_value_end(value);
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::push_back(Thread g, const_reference value)
	{
		if (size() + g.size() < capacity())
		{
			alloc_traits::construct(get_allocator(), to_pointer(m_end) + g.thread_rank(), value);
			advance(m_end, g.size());
		}
		else
			insert_value_end(g, value);
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::push_back(value_type&& value)
	{
		if (m_end != m_end_capacity)
		{
			alloc_traits::construct(get_allocator(), to_pointer(m_end), std::move(value));
			++m_end;
		}
		else
			insert_value_end(std::move(value));
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::push_back(Thread g, value_type&& value)
	{
		if (size() + g.size() < capacity())
		{
			alloc_traits::construct(get_allocator(), to_pointer(m_end) + g.thread_rank(), std::move(value));
			advance(m_end, g.size());
		}
		else
			insert_value_end(g, std::move(value));
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::reserve(size_type n)
	{
		if (n > capacity())
		{
			pointer ptr = allocate(n);
			auto end_ptr = uninitialized_move(m_begin, m_end, ptr);

			m_begin = ptr;
			m_end = ptr + n;
			m_end_capacity = ptr + n;
		}
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::reserve(Thread g, size_type n)
	{
		if (n > capacity())
		{
			pointer ptr = allocate(g, n);
			auto end_ptr = uninitialized_move(g, m_begin, m_end, ptr);

			if (g.thread_rank() == 0)
			{
				m_begin = ptr;
				m_end = ptr + n;
				m_end_capacity = ptr + n;
			}
		}
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::shrink_to_fit()
	{
		if (capacity() > size())
		{
			vector temp(move_iterator<iterator>(begin()), move_iterator<iterator>(end()), get_allocator());
			swap(temp);
		}
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::shrink_to_fit(Thread g)
	{
		if (capacity() > size())
		{
			vector temp(g, move_iterator<iterator>(begin()), move_iterator<iterator>(end()), get_allocator());
			swap(temp);
		}
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::size_type vector<T, Allocator>::size() const noexcept
	{
		return static_cast<size_type>(distance(m_begin, m_end));
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::swap(vector& other)
	{
		std::swap(m_begin, other.m_begin);
		std::swap(m_end, other.m_end);
		std::swap(m_end_capacity, other.m_end_capacity);
		std::swap(m_allocator, other.m_allocator);
	}

	template <typename T, class Allocator>
	GPU_DEVICE typename vector<T, Allocator>::pointer vector<T, Allocator>::allocate(size_type n)
	{
		return alloc_traits::allocate(get_allocator(), n);
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE typename vector<T, Allocator>::pointer vector<T, Allocator>::allocate(Thread g, size_type n)
	{
		return alloc_traits::allocate(g, get_allocator(), n);
	}

	template <typename T, class Allocator>
	GPU_DEVICE void vector<T, Allocator>::deallocate(pointer ptr)
	{
		alloc_traits::deallocate(get_allocator(), ptr, capacity());
	}

	template <typename T, class Allocator>
	template <class Thread>
	GPU_DEVICE void vector<T, Allocator>::deallocate(Thread g, pointer ptr)
	{
		alloc_traits::deallocate(g, get_allocator(), ptr, capacity());
	}

	template <typename T, class Allocator>
	template <typename... Args>
	GPU_DEVICE void vector<T, Allocator>::insert_value_end(Args&&... args)
	{
		const size_type old_size = max(size(), static_cast<size_type>(DEFAULT_BYTE_ALIGNMENT / (2 * sizeof(T))));
		const size_type new_size = old_size * 2;
		pointer const ptr = allocate(new_size);

		pointer new_end = uninitialized_move(m_begin, m_end, ptr);
		alloc_traits::construct(get_allocator(), to_pointer(new_end), std::forward<Args>(args)...);
		++new_end;

		deallocate(m_begin);

		m_begin = ptr;
		m_end = new_end;
		m_end_capacity = m_begin + new_size;
	}

	template <typename T, class Allocator>
	template <typename... Args>
	GPU_DEVICE void vector<T, Allocator>::insert_value_end(block_t g, Args&&... args)
	{
		const size_type old_size = max(size(), static_cast<size_type>(g.size() / 2));
		const size_type new_size = old_size * 2;
		pointer const ptr = allocate(g, new_size);

		pointer new_end = uninitialized_move(g, m_begin, m_end, ptr);
		alloc_traits::construct(get_allocator(), to_pointer(new_end) + g.thread_rank(), std::forward<Args>(args)...);
		advance(new_end, g.size());

		deallocate(g, m_begin);

		if (g.thread_rank() == 0)
		{
			m_begin = ptr;
			m_end = new_end;
			m_end_capacity = m_begin + new_size;
		}
		g.sync();
	}

	template <typename T, class Allocator>
	template <typename... Args, unsigned int tile_sz>
	GPU_DEVICE void vector<T, Allocator>::insert_value_end(block_tile_t<tile_sz> g, Args&&... args)
	{
		const size_type old_size = size();
		const size_type new_size = old_size * 2;
		pointer const ptr = allocate(g, new_size);

		pointer new_end = uninitialized_move(g, m_begin, m_end, ptr);
		alloc_traits::construct(get_allocator(), to_pointer(new_end) + g.thread_rank(), std::forward<Args>(args)...);
		advance(new_end, g.size());

		destroy(g, m_begin, m_end);
		deallocate(g, m_begin);

		if (g.thread_rank() == 0)
		{
			m_begin = ptr;
			m_end = new_end;
			m_end_capacity = m_begin + new_size;
		}
		g.sync();
	}
}
