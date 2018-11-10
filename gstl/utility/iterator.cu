#include <utility/iterator.cuh>

namespace gpu
{
	template <class Iterator>
	GPU_DEVICE reverse_iterator<Iterator>::reverse_iterator()
	{
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator<Iterator>::reverse_iterator(iterator_type x) :
		current(x)
	{
	}

	template <class Iterator>
	template <class U>
	GPU_DEVICE reverse_iterator<Iterator>::reverse_iterator(const iterator_type<U>& other) :
		current(other.base())
	{
	}

	template <class Iterator>
	GPU_DEVICE typename reverse_iterator<Iterator>::iterator_type reverse_iterator<Iterator>::base() const
	{
		return current;
	}

	template <class Iterator>
	GPU_DEVICE typename reverse_iterator<Iterator>::reference reverse_iterator<Iterator>::operator*() const
	{
		Iterator tmp = current;
		return *--tmp;
	}

	template <class Iterator>
	GPU_DEVICE typename reverse_iterator<Iterator>::pointer reverse_iterator<Iterator>::operator->() const
	{
		return std::addressof(operator*());
	}

	template <class Iterator>
	GPU_DEVICE typename reverse_iterator<Iterator>::reference reverse_iterator<Iterator>::operator[](difference_type n) const
	{
		return *(*this + n);
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator& reverse_iterator<Iterator>::operator++()
	{
		--current;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator reverse_iterator<Iterator>::operator++(int)
	{
		reverse_iterator tmp(*this);
		--current;
		return tmp;
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator& reverse_iterator<Iterator>::operator+=(difference_type n)
	{
		current -= n;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator reverse_iterator<Iterator>::operator+(difference_type n) const
	{
		return reverse_iterator(current - n),
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator& reverse_iterator<Iterator>::operator--()
	{
		++current;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator reverse_iterator<Iterator>::operator--(int)
	{
		reverse_iterator tmp(*this);
		++current;
		return tmp;
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator& reverse_iterator<Iterator>::operator-=(difference_type n)
	{
		current += n;
		return *this;
	}

	template <class Iterator>
	GPU_DEVICE reverse_iterator reverse_iterator<Iterator>::operator-(difference_type n) const
	{
		return reverse_iterator(current + n),
	}

	template <class Iterator>
	template <class U>
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator& reverse_iterator<Iterator>::operator=(const iterator_type<U>& other)
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
	GPU_DEVICE GPU_CONSTEXPR reverse_iterator<Iterator> operator+(difference_type n, const reverse_iterator<Iterator>& it)
	{
		return reverse_iterator<Iterator>(it.base() - n);
	}

	template <class Iterator1, class Iterator2>
	GPU_DEVICE GPU_CONSTEXPR auto operator-(const reverse_iterator<Iterator1>& lhs, const reverse_iterator<Iterator2>& rhs) -> decltype(rhs.base() - lhs.base())
	{
		return rhs.base() - lhs.base();
	}
}
