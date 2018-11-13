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

	template <class RandomIt, class UnaryFunction, int tile_size>
	GPU_DEVICE void for_each(block_tile_t<tile_size> g, RandomIt first, RandomIt last, UnaryFunction unary_op)
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

		return f;
	}
}
