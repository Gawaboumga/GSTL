#include <gstl/allocators/allocated_memory.cuh>

namespace gpu
{
	template <typename T>
	GPU_DEVICE allocated_memory<T>::iterator allocated_memory<T>::begin() noexcept
	{
		return iterator(m_start);
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::const_iterator allocated_memory<T>::begin() const noexcept
	{
		return const_iterator(m_start);
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::const_iterator allocated_memory<T>::cbegin() const noexcept
	{
		return const_iterator(m_start);
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::iterator allocated_memory<T>::end() noexcept
	{
		return iterator(m_end);
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::const_iterator allocated_memory<T>::end() const noexcept
	{
		return const_iterator(m_end);
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::const_iterator allocated_memory<T>::cend() const noexcept
	{
		return const_iterator(m_end);
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::reverse_iterator allocated_memory<T>::rbegin() noexcept
	{
		return reverse_iterator(end());
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::const_reverse_iterator allocated_memory<T>::rbegin() const noexcept
	{
		return const_reverse_iterator(end());
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::const_reverse_iterator allocated_memory<T>::crbegin() const noexcept
	{
		return const_reverse_iterator(end());
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::reverse_iterator allocated_memory<T>::rend() noexcept
	{
		return reverse_iterator(begin());
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::const_reverse_iterator allocated_memory<T>::rend() const noexcept
	{
		return const_reverse_iterator(begin());
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::const_reverse_iterator allocated_memory<T>::crend() const noexcept
	{
		return const_reverse_iterator(begin());
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::allocated_memory(std::nullptr_t) noexcept :
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		m_start{ nullptr },
		m_end{ nullptr }
	#else
		m_ptr{ nullptr }
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	{
	}


	template <typename T>
	GPU_DEVICE allocated_memory<T>::allocated_memory(block_t g, T* ptr, size_type count) noexcept :
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		m_start{ ptr },
		m_end{ ptr + count }
	#else
		m_ptr{ ptr }
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	{
		post_condition();
	}

	template <typename T>
	template <class BlockTile>
	GPU_DEVICE allocated_memory<T>::allocated_memory(BlockTile g, T* ptr, size_type count) noexcept :
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		m_start{ ptr },
		m_end{ ptr + count }
	#else
		m_ptr{ ptr }
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	{
		post_condition();
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::allocated_memory(T* ptr, size_type count) noexcept :
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		m_start{ ptr },
		m_end{ ptr + count + 1 }
	#else
		m_ptr{ ptr + 1 }
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	{
		post_condition();
	}

	template <typename T>
	GPU_DEVICE T* allocated_memory<T>::data() noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return m_start;
	#else
		return m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE const T* allocated_memory<T>::data() const noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return m_start;
	#else
		return m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE bool allocated_memory<T>::is_valid() const noexcept
	{
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return m_start != nullptr;
	#else
		return m_ptr != nullptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE typename allocated_memory<T>::reference allocated_memory<T>::operator[](size_type pos) noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		#if defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(m_start + pos < m_end);
		#endif // GPU_DEBUG_OUT_OF_RANGE
		return *(m_start + pos);
	#else
		return *(m_ptr + pos);
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE typename allocated_memory<T>::const_reference allocated_memory<T>::operator[](size_type pos) const noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		#if defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(m_start + pos < m_end);
		#endif // GPU_DEBUG_OUT_OF_RANGE
		return *(m_start + pos);
	#else
		return *(m_ptr + pos);
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T> allocated_memory<T>::operator+(size_type pos) const noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return allocated_memory(m_start + pos, m_end);
	#else
		return allocated_memory(m_ptr + pos);
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>& allocated_memory<T>::operator+=(size_type pos) noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		m_start += pos;
		return *this;
	#else
		m_ptr += pos;
		return *this;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>& allocated_memory<T>::operator++() noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		#if defined(GPU_DEBUG_OUT_OF_RANGE)
			ENSURE(m_start + 1 < m_end);
		#endif // GPU_DEBUG_OUT_OF_RANGE
		++m_start;
		return *this;
	#else
		++m_ptr;
		return *this;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T> allocated_memory<T>::operator++(int) noexcept
	{
		allocated_memory tmp(*this);
		++(*this);
		return tmp;
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T> allocated_memory<T>::operator-(size_type pos) const noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return allocated_memory(m_start - pos, m_end);
	#else
		return allocated_memory(m_ptr - pos);
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>& allocated_memory<T>::operator-=(size_type pos) noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		m_start -= pos;
		return *this;
	#else
		m_ptr -= pos;
		return *this;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>& allocated_memory<T>::operator--() noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		--m_start;
		return *this;
	#else
		--m_ptr;
		return *this;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T> allocated_memory<T>::operator--(int) noexcept
	{
		allocated_memory tmp(*this);
		--(*this);
		return tmp;
	}

	template <typename T>
	GPU_DEVICE bool allocated_memory<T>::operator==(const allocated_memory& rhs) const noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return m_start == rhs.m_start;
	#else
		return m_ptr == rhs.m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE bool allocated_memory<T>::operator!=(const allocated_memory& rhs) const noexcept
	{
		return !operator==(rhs);
	}

	template <typename T>
	GPU_DEVICE T& allocated_memory<T>::operator*() noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return *m_start;
	#else
		return *m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE const T& allocated_memory<T>::operator*() const noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return *m_start;
	#else
		return *m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE T* allocated_memory<T>::operator->() noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return m_start;
	#else
		return m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE const T* allocated_memory<T>::operator->() const noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return m_start;
	#else
		return m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE bool allocated_memory<T>::operator<(const allocated_memory& rhs) const noexcept
	{
		post_condition();

	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return m_start < rhs.m_start;
	#else
		return m_ptr < rhs.m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE void allocated_memory<T>::debug() const
	{
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		printf("allocated_memory: (%p, %p)", m_start, m_end);
	#else
		printf("allocated_memory: (%p)", m_ptr);
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE void allocated_memory<T>::invalidate() noexcept
	{
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		m_start = nullptr;
		m_end = nullptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE allocated_memory<T>::allocated_memory(T* start, T* end) noexcept :
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		m_start{ start },
		m_end{ end }
	#else
		m_ptr{ start }
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	{
		post_condition();
	}

	template <typename T>
	GPU_DEVICE void allocated_memory<T>::post_condition() const noexcept
	{
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		if (m_start)
			ENSURE(m_end != nullptr);
		else
			ENSURE(m_end == nullptr);
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR typename allocated_memory<T>::difference_type distance(const allocated_memory<T>& first, const allocated_memory<T>& last)
	{
	#if defined(GPU_DEBUG_ALLOCATED_MEMORY)
		return last.m_start - first.m_start;
	#else
		return last.m_ptr - first.m_ptr;
	#endif // GPU_DEBUG_ALLOCATED_MEMORY
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T* to_pointer(allocated_memory<T>& memory)
	{
		return memory.data();
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR const T* to_pointer(const allocated_memory<T>& memory)
	{
		return memory.data();
	}
}
