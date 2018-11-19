#include "partial_sum.cuh"

#include <iterator>

namespace gpu
{
	template <class InputIt, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt partial_sum(InputIt first, InputIt last, OutputIt d_first)
	{
		return partial_sum(first, last, d_first, plus<>());
	}

	template< class InputIt, class OutputIt, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt partial_sum(InputIt first, InputIt last, OutputIt d_first, BinaryOperation op)
	{
		if (first == last)
			return d_first;

		typename std::iterator_traits<InputIt>::value_type sum = *first;
		*d_first = sum;

		while (++first != last)
		{
			sum = op(std::move(sum), *first); // std::move since C++20
			*++d_first = sum;
		}
		return ++d_first;
	}
}
