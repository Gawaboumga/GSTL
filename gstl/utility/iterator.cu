#include <gstl/utility/iterator.cuh>

namespace gpu
{
	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator>::move_iterator()
	{
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator>::move_iterator(iterator_type x) :
		current(x)
	{
	}

	template <class Iterator>
	template <class U>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator>::move_iterator(const move_iterator<U>& other) :
		current(other.base())
	{
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR typename move_iterator<Iterator>::iterator_type move_iterator<Iterator>::base() const
	{
		return current;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR typename move_iterator<Iterator>::reference move_iterator<Iterator>::operator*() const
	{
		return *current;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR typename move_iterator<Iterator>::pointer move_iterator<Iterator>::operator->() const
	{
		return current;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR typename move_iterator<Iterator>::reference move_iterator<Iterator>::operator[](difference_type n) const
	{
		return *(*this + n);
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator>& move_iterator<Iterator>::operator++()
	{
		++current;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator> move_iterator<Iterator>::operator++(int)
	{
		move_iterator tmp(*this);
		++current;
		return tmp;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator>& move_iterator<Iterator>::operator+=(difference_type n)
	{
		current += n;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator> move_iterator<Iterator>::operator+(difference_type n) const
	{
		return move_iterator(current + n);
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator>& move_iterator<Iterator>::operator--()
	{
		--current;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator> move_iterator<Iterator>::operator--(int)
	{
		move_iterator tmp(*this);
		--current;
		return tmp;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator>& move_iterator<Iterator>::operator-=(difference_type n)
	{
		current -= n;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator> move_iterator<Iterator>::operator-(difference_type n) const
	{
		return move_iterator(current - n);
	}

	template <class Iterator>
	template <class U>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator>& move_iterator<Iterator>::operator=(const move_iterator<U>& other)
	{
		current = other.base();
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator> make_move_iterator(Iterator it)
	{
		return move_iterator<Iterator>(it);
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator==(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs)
	{
		return lhs.base() == rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator!=(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs)
	{
		return lhs.base() != rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator<(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs)
	{
		return lhs.base() < rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator<=(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs)
	{
		return lhs.base() <= rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator>(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs)
	{
		return lhs.base() > rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator>=(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs)
	{
		return lhs.base() >= rhs.base();
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR move_iterator<Iterator> operator+(typename move_iterator<Iterator>::difference_type n, const move_iterator<Iterator>& it)
	{
		return move_iterator<Iterator>(it.base() + n);
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR auto operator-(const move_iterator<Iterator1>& lhs, const move_iterator<Iterator2>& rhs) -> decltype(lhs.base() - rhs.base())
	{
		return lhs.base() - rhs.base();
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator>::reverse_iterator()
	{
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator>::reverse_iterator(iterator_type x) :
		current(x)
	{
	}

	template <class Iterator>
	template <class U>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator>::reverse_iterator(const reverse_iterator<U>& other) :
		current(other.base())
	{
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR typename reverse_iterator<Iterator>::iterator_type reverse_iterator<Iterator>::base() const
	{
		return current;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR typename reverse_iterator<Iterator>::reference reverse_iterator<Iterator>::operator*() const
	{
		Iterator tmp = current;
		return *--tmp;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR typename reverse_iterator<Iterator>::pointer reverse_iterator<Iterator>::operator->() const
	{
		return std::addressof(operator*());
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR typename reverse_iterator<Iterator>::reference reverse_iterator<Iterator>::operator[](difference_type n) const
	{
		return *(*this + n);
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator>& reverse_iterator<Iterator>::operator++()
	{
		--current;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> reverse_iterator<Iterator>::operator++(int)
	{
		reverse_iterator tmp(*this);
		--current;
		return tmp;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator>& reverse_iterator<Iterator>::operator+=(difference_type n)
	{
		current -= n;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> reverse_iterator<Iterator>::operator+(difference_type n) const
	{
		return reverse_iterator(current - n);
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator>& reverse_iterator<Iterator>::operator--()
	{
		++current;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> reverse_iterator<Iterator>::operator--(int)
	{
		reverse_iterator tmp(*this);
		++current;
		return tmp;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator>& reverse_iterator<Iterator>::operator-=(difference_type n)
	{
		current += n;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> reverse_iterator<Iterator>::operator-(difference_type n) const
	{
		return reverse_iterator(current + n);
	}

	template <class Iterator>
	template <class U>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator>& reverse_iterator<Iterator>::operator=(const reverse_iterator<U>& other)
	{
		current = other.base();
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> make_reverse_iterator(Iterator it)
	{
		return reverse_iterator<Iterator>(it);
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator==(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs)
	{
		return lhs.base() == rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator!=(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs)
	{
		return lhs.base() != rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator<(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs)
	{
		return lhs.base() < rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator<=(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs)
	{
		return lhs.base() <= rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator>(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs)
	{
		return lhs.base() > rhs.base();
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR bool operator>=(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs)
	{
		return lhs.base() >= rhs.base();
	}

	template <class Iterator>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> operator+(typename reverse_iterator<Iterator>::difference_type n, const reverse_iterator<Iterator>& it)
	{
		return reverse_iterator<Iterator>(it.base() - n);
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR auto operator-(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs) -> decltype(rhs.base() - lhs.base())
	{
		return rhs.base() - lhs.base();
	}

	namespace
	{
		template <class InputIt>
		inline GPU_DEVICE GPU_CONSTEXPR void advance(InputIt& it, typename std::iterator_traits<InputIt>::difference_type n, std::input_iterator_tag)
		{
			for (; n > 0; --n)
				++it;
		}

		template <class BiDirIt>
		inline GPU_DEVICE GPU_CONSTEXPR void advance(BiDirIt& it, typename std::iterator_traits<BiDirIt>::difference_type n, std::bidirectional_iterator_tag)
		{
			if (n >= 0)
				for (; n > 0; --n)
					++i;
			else
				for (; n < 0; ++n)
					--i;
		}

		template <class RandomIt>
		inline GPU_DEVICE GPU_CONSTEXPR void advance(RandomIt& it, typename std::iterator_traits<RandomIt>::difference_type n, std::random_access_iterator_tag)
		{
			it += n;
		}

		template <class InputIt>
		inline GPU_DEVICE GPU_CONSTEXPR typename std::iterator_traits<InputIt>::difference_type distance(InputIt first, InputIt last, std::input_iterator_tag)
		{
			typename std::iterator_traits<InputIt>::difference_type r(0);
			for (; first != last; ++first)
				++r;
			return r;
		}

		template <class RandomIt>
		inline GPU_DEVICE GPU_CONSTEXPR typename std::iterator_traits<RandomIt>::difference_type distance(RandomIt first, RandomIt last, std::random_access_iterator_tag)
		{
			return last - first;
		}
	}

	template <class InputIt>
	GPU_DEVICE GPU_CONSTEXPR void advance(InputIt& it, typename std::iterator_traits<InputIt>::difference_type n)
	{
		return advance(it, n, typename std::iterator_traits<InputIt>::iterator_category());
	}

	template <class InputIt>
	GPU_DEVICE GPU_CONSTEXPR typename std::iterator_traits<InputIt>::difference_type distance(InputIt first, InputIt last)
	{
		return distance(first, last, typename std::iterator_traits<InputIt>::iterator_category());
	}

	template <class InputIt>
	GPU_DEVICE GPU_CONSTEXPR InputIt next(InputIt x, typename std::iterator_traits<InputIt>::difference_type n)
	{
		advance(x, n);
		return x;
	}
}
