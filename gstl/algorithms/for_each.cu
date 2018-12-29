#include <gstl/algorithms/for_each.cuh>

namespace gpu
{
	template <class RandomIt, class UnaryFunction>
	GPU_DEVICE void for_each(block_t g, RandomIt first, RandomIt last, UnaryFunction unary_op)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		while (offset + thid < len)
		{
			unary_op(*(first + offset + thid));
			offset += g.size();
		}
	}

	template <class BlockTile, class RandomIt, class UnaryFunction>
	GPU_DEVICE void for_each(BlockTile g, RandomIt first, RandomIt last, UnaryFunction unary_op)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		while (offset + thid < len)
		{
			unary_op(*(first + offset + thid));
			offset += g.size();
		}
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
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		while (offset + thid < n)
		{
			unary_op(*(first + offset + thid));
			offset += g.size();
		}

		return first + n;
	}

	template <class BlockTile, class ForwardIt, class Size, class UnaryFunction>
	GPU_DEVICE ForwardIt for_each_n(BlockTile g, ForwardIt first, Size n, UnaryFunction unary_op)
	{
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		while (offset + thid < n)
		{
			unary_op(*(first + offset + thid));
			offset += g.size();
		}

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
