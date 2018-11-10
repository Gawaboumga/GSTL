#pragma once

#ifndef GPU_CONTAINERS_ARRAY_HPP
#define GPU_CONTAINERS_ARRAY_HPP

#include <prerequisites.hpp>
#include <utility/iterator.cuh>

namespace gpu
{
	template <typename T, unsigned int N>
	struct array
	{
		using value_type = T;
		using size_type = size_t;
		using difference_type = ptrdiff_t;
		using reference = value_type&;
		using const_reference = const value_type&;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using iterator = value_type*;
		using const_iterator = const value_type*;
		using reverse_iterator = reverse_iterator<iterator>;
		using const_reverse_iterator = reverse_iterator<const_iterator>;

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

		GPU_DEVICE GPU_CONSTEXPR reference back();
		GPU_DEVICE GPU_CONSTEXPR const_reference back() const;

		GPU_DEVICE GPU_CONSTEXPR value_type* data();
		GPU_DEVICE GPU_CONSTEXPR const value_type* data() const;

		GPU_DEVICE GPU_CONSTEXPR bool empty() const;

		GPU_DEVICE GPU_CONSTEXPR reference front();
		GPU_DEVICE GPU_CONSTEXPR const_reference front() const;

		GPU_DEVICE GPU_CONSTEXPR reference operator[](size_type n);
		GPU_DEVICE GPU_CONSTEXPR const_reference operator[](size_type n) const;

		GPU_DEVICE GPU_CONSTEXPR size_type size() const;

		value_type m_elems[N];
	};
}

#include <containers/array.cu>

#endif // GPU_CONTAINERS_ARRAY_HPP
