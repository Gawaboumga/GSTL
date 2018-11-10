#pragma once

#ifndef GPU_UTILITY_ITERATOR_HPP
#define GPU_UTILITY_ITERATOR_HPP

#include <prerequisites.hpp>

#include <iterator>

namespace gpu
{
	template <class Iterator>
	class reverse_iterator
	{
		public:
			using value_type = std::iterator_traits<Iterator>::value_type;
			using difference_type = std::iterator_traits<Iterator>::difference_type;
			using pointer = std::iterator_traits<Iterator>::pointer;
			using reference = std::iterator_traits<Iterator>::reference;
			using iterator_category = std::iterator_traits<Iterator>::iterator_category;
			using iterator_type = Iterator;

		public:
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator();
			GPU_DEVICE GPU_CONSTEXPR explicit reverse_iterator(iterator_type x);
			template <class U>
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator(const iterator_type<U>& other);

			GPU_DEVICE GPU_CONSTEXPR iterator_type base() const;

			GPU_DEVICE GPU_CONSTEXPR reference operator*() const;
			GPU_DEVICE GPU_CONSTEXPR pointer operator->() const;

			GPU_DEVICE GPU_CONSTEXPR reference operator[](ifference_type n) const;

			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator++();
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator operator++(int);
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator+=(difference_type n);
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator operator+(difference_type n) const;

			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator--();
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator operator--(int);
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator-=(difference_type n);
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator operator-(difference_type n) const;

			template <class U>
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator=(const iterator_type<U>& other);

		protected:
			iterator_type current;
	};

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> make_reverse_iterator(Iterator it);

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator==(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator!=(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator<(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator<=(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator>(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator>=(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs);

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> operator+(typename reverse_iterator<Iterator>::difference_type n, const reverse_iterator<Iterator>& it);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR auto operator-(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs) -> decltype(rhs.base() - lhs.base());
}

#include <utility/iterator.cu>

#endif // GPU_UTILITY_ITERATOR_HPP
