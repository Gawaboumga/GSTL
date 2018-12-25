#include <gstl/containers/array.cuh>

namespace gpu
{
	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::iterator array<T, N>::begin()
	{
		return iterator(data());
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_iterator array<T, N>::begin() const
	{
		return const_iterator(data());
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_iterator array<T, N>::cbegin() const
	{
		return begin();
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::iterator array<T, N>::end()
	{
		return iterator(data() + N);
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_iterator array<T, N>::end() const
	{
		return const_iterator(data() + N);
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_iterator array<T, N>::cend() const
	{
		return end();
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::reverse_iterator array<T, N>::rbegin()
	{
		return reverse_iterator(end());
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_reverse_iterator array<T, N>::rbegin() const
	{
		return const_reverse_iterator(end());
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_reverse_iterator array<T, N>::crbegin() const
	{
		return rbegin();
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::reverse_iterator array<T, N>::rend()
	{
		return reverse_iterator(begin());
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_reverse_iterator array<T, N>::rend() const
	{
		return const_reverse_iterator(begin());
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_reverse_iterator array<T, N>::crend() const
	{
		return rend();
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::reference array<T, N>::back()
	{
		return m_elems[N - 1];
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_reference array<T, N>::back() const
	{
		return m_elems[N - 1];
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::value_type* array<T, N>::data()
	{
		return m_elems;
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR const typename array<T, N>::value_type* array<T, N>::data() const
	{
		return m_elems;
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR bool array<T, N>::empty() const
	{
		return false;
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::reference array<T, N>::front()
	{
		return m_elems[0];
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_reference array<T, N>::front() const
	{
		return m_elems[0];
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::reference array<T, N>::operator[](size_type n)
	{
	#if defined(GPU_DEBUG_ARRAY) || defined(GPU_DEBUG_OUT_OF_RANGE)
		ENSURE(n < size());
	#endif // GPU_DEBUG_ARRAY
		return m_elems[n];
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::const_reference array<T, N>::operator[](size_type n) const
	{
	#if defined(GPU_DEBUG_ARRAY) || defined(GPU_DEBUG_OUT_OF_RANGE)
		ENSURE(n < size());
	#endif // GPU_DEBUG_ARRAY
		return m_elems[n];
	}

	template <typename T, unsigned int N>
	GPU_DEVICE GPU_CONSTEXPR typename array<T, N>::size_type array<T, N>::size() const
	{
		return N;
	}
}
