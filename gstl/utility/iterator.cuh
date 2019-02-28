#pragma once

#ifndef GPU_UTILITY_ITERATOR_HPP
#define GPU_UTILITY_ITERATOR_HPP

#include <gstl/prerequisites.hpp>

#include <iterator>

namespace gpu
{
	template <class Iterator>
	class move_iterator
	{
		public:
			using iterator_type = Iterator;
			using iterator_category = typename std::iterator_traits<iterator_type>::iterator_category;
			using value_type = typename std::iterator_traits<iterator_type>::value_type;
			using difference_type = typename std::iterator_traits<iterator_type>::difference_type;
			using pointer = iterator_type;
			using reference = typename std::iterator_traits<iterator_type>::reference;

			GPU_DEVICE GPU_CONSTEXPR move_iterator();
			GPU_DEVICE GPU_CONSTEXPR explicit move_iterator(iterator_type x);
			template <class U>
			GPU_DEVICE GPU_CONSTEXPR move_iterator(const move_iterator<U>& other);

			GPU_DEVICE GPU_CONSTEXPR iterator_type base() const;

			GPU_DEVICE GPU_CONSTEXPR reference operator*() const;
			GPU_DEVICE GPU_CONSTEXPR pointer operator->() const;

			GPU_DEVICE GPU_CONSTEXPR reference operator[](difference_type n) const;

			GPU_DEVICE GPU_CONSTEXPR move_iterator& operator++();
			GPU_DEVICE GPU_CONSTEXPR move_iterator operator++(int);
			GPU_DEVICE GPU_CONSTEXPR move_iterator& operator+=(difference_type n);
			GPU_DEVICE GPU_CONSTEXPR move_iterator operator+(difference_type n) const;

			GPU_DEVICE GPU_CONSTEXPR move_iterator& operator--();
			GPU_DEVICE GPU_CONSTEXPR move_iterator operator--(int);
			GPU_DEVICE GPU_CONSTEXPR move_iterator& operator-=(difference_type n);
			GPU_DEVICE GPU_CONSTEXPR move_iterator operator-(difference_type n) const;

			template <class U>
			GPU_DEVICE GPU_CONSTEXPR move_iterator& operator=(const move_iterator<U>& other);

		protected:
			iterator_type current;
	};

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator> make_reverse_iterator(Iterator it);

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator==(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator!=(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator<(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator<=(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator>(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator>=(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs);

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator> operator+(typename move_iterator<Iterator>::difference_type n, const move_iterator<Iterator>& it);
	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR auto operator-(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs) -> decltype(lhs.base() - rhs.base());

	template <class Iterator>
	class reverse_iterator
	{
		public:
			using value_type = typename std::iterator_traits<Iterator>::value_type;
			using difference_type = typename std::iterator_traits<Iterator>::difference_type;
			using pointer = typename std::iterator_traits<Iterator>::pointer;
			using reference = typename std::iterator_traits<Iterator>::reference;
			using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;
			using iterator_type = Iterator;

		public:
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator();
			GPU_DEVICE GPU_CONSTEXPR explicit reverse_iterator(iterator_type x);
			template <class U>
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator(const reverse_iterator<U>& other);

			GPU_DEVICE GPU_CONSTEXPR iterator_type base() const;

			GPU_DEVICE GPU_CONSTEXPR reference operator*() const;
			GPU_DEVICE GPU_CONSTEXPR pointer operator->() const;

			GPU_DEVICE GPU_CONSTEXPR reference operator[](difference_type n) const;

			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator++();
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator operator++(int);
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator+=(difference_type n);
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator operator+(difference_type n) const;

			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator--();
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator operator--(int);
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator-=(difference_type n);
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator operator-(difference_type n) const;

			template <class U>
			GPU_DEVICE GPU_CONSTEXPR reverse_iterator& operator=(const reverse_iterator<U>& other);

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

	// Other

	template <class InputIt>
	GPU_DEVICE GPU_CONSTEXPR void advance(InputIt& i, typename std::iterator_traits<InputIt>::difference_type n);

	template <class InputIt>
	GPU_DEVICE GPU_CONSTEXPR typename std::iterator_traits<InputIt>::difference_type distance(InputIt first, InputIt last);

	template <class InputIt>  // constexpr in C++17
	GPU_DEVICE GPU_CONSTEXPR InputIt next(InputIt x, typename std::iterator_traits<InputIt>::difference_type n = 1);

	template <class BidirIt>
	GPU_DEVICE GPU_CONSTEXPR BidirIt prev(BidirIt it, typename std::iterator_traits<BidirIt>::difference_type n = 1);
}

#include <gstl/utility/iterator.cu>

#endif // GPU_UTILITY_ITERATOR_HPP
