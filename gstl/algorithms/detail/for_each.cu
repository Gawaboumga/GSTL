#include <gstl/algorithms/detail/for_each.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class UnaryFunction>
		GPU_DEVICE void for_each(Thread g, RandomIt first, RandomIt last, UnaryFunction unary_op)
		{
			for_each_n(g, first, distance(first, last), unary_op);
		}

		template <class Thread, class ForwardIt, class Size, class UnaryFunction>
		GPU_DEVICE void for_each_n(Thread g, ForwardIt first, Size n, UnaryFunction unary_op)
		{
			offset_t thid = g.thread_rank();

			while (thid < n)
			{
				unary_op(*(first + thid));
				thid += g.size();
			}
		}
	}
}
