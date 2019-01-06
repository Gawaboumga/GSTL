#include <gstl/algorithms/detail/transform.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt1, class RandomIt2, class UnaryOperation>
		GPU_DEVICE void transform(Thread g, RandomIt1 first, RandomIt1 last, RandomIt2 d_first, UnaryOperation unary_op)
		{
			offset_t len = distance(first, last);
			offset_t thid = g.thread_rank();

			while (thid < len)
			{
				*(d_first + thid) = unary_op(*(first + thid));
				thid += g.size();
			}
		}

		template <class Thread, class RandomIt1, class RandomIt2, class RandomIt3, class BinaryOperation>
		GPU_DEVICE void transform(Thread g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt3 d_first, BinaryOperation binary_op)
		{
			offset_t len = distance(first1, last1);
			offset_t thid = g.thread_rank();

			while (thid < len)
			{
				*(d_first + thid) = binary_op(*(first1 + thid), *(first2 + thid));
				thid += g.size();
			}
		}
	}
}
