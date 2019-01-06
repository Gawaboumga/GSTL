#include <gstl/algorithms/transform.cuh>

#include <gstl/algorithms/detail/transform.cuh>

namespace gpu
{
	template <class RandomIt1, class RandomIt2, class UnaryOperation>
	GPU_DEVICE RandomIt2 transform(block_t g, RandomIt1 first, RandomIt1 last, RandomIt2 d_first, UnaryOperation unary_op)
	{
		detail::transform(g, first, last, d_first, unary_op);
		return d_first + distance(first, last);
	}

	template <class BlockTile, class RandomIt1, class RandomIt2, class UnaryOperation>
	GPU_DEVICE RandomIt2 transform(BlockTile g, RandomIt1 first, RandomIt1 last, RandomIt2 d_first, UnaryOperation unary_op)
	{
		detail::transform(g, first, last, d_first, unary_op);
		return d_first + distance(first, last);
	}

	template <class ForwardIt, class OutputIt, class UnaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform(ForwardIt first, ForwardIt last, OutputIt d_first, UnaryOperation unary_op)
	{
		while (first != last)
			*d_first++ = unary_op(*first++);

		return d_first;
	}

	template <class RandomIt1, class RandomIt2, class RandomIt3, class BinaryOperation>
	GPU_DEVICE RandomIt3 transform(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt3 d_first, BinaryOperation binary_op)
	{
		detail::transform(g, first1, last1, first2, d_first, binary_op);
		return d_first + distance(first1, last1);
	}

	template <class BlockTile, class RandomIt1, class RandomIt2, class RandomIt3, class BinaryOperation>
	GPU_DEVICE RandomIt3 transform(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt3 d_first, BinaryOperation binary_op)
	{
		detail::transform(g, first1, last1, first2, d_first, binary_op);
		return d_first + distance(first1, last1);
	}

	template <class ForwardIt1, class ForwardIt2, class OutputIt, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, OutputIt d_first, BinaryOperation binary_op)
	{
		while (first1 != last1)
			*d_first++ = binary_op(*first1++, *first2++);

		return d_first;
	}
}
