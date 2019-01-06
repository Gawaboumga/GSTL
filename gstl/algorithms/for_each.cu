#include <gstl/algorithms/for_each.cuh>

#include <gstl/algorithms/detail/for_each.cuh>

namespace gpu
{
	template <class RandomIt, class UnaryFunction>
	GPU_DEVICE void for_each(block_t g, RandomIt first, RandomIt last, UnaryFunction unary_op)
	{
		detail::for_each(g, first, last, unary_op);
	}

	template <class BlockTile, class RandomIt, class UnaryFunction>
	GPU_DEVICE void for_each(BlockTile g, RandomIt first, RandomIt last, UnaryFunction unary_op)
	{
		detail::for_each(g, first, last, unary_op);
	}

	template <class ForwardIt, class UnaryFunction>
	GPU_DEVICE GPU_CONSTEXPR void for_each(ForwardIt first, ForwardIt last, UnaryFunction unary_op)
	{
		for (; first != last; ++first)
			unary_op(*first);
	}

	template <class ForwardIt, class Size, class UnaryFunction>
	GPU_DEVICE ForwardIt for_each_n(block_t g, ForwardIt first, Size n, UnaryFunction unary_op)
	{
		detail::for_each_n(g, first, n, unary_op);
		return first + n;
	}

	template <class BlockTile, class ForwardIt, class Size, class UnaryFunction>
	GPU_DEVICE ForwardIt for_each_n(BlockTile g, ForwardIt first, Size n, UnaryFunction unary_op)
	{
		detail::for_each_n(g, first, n, unary_op);
		return first + n;
	}

	template <class InputIt, class Size, class UnaryFunction>
	GPU_DEVICE GPU_CONSTEXPR InputIt for_each_n(InputIt first, Size n, UnaryFunction unary_op)
	{
		for (Size i = 0; i < n; ++first, (void)++i)
			unary_op(*first);

		return first;
	}
}
