#include <gstl/algorithms/detail/enumerate.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class Function>
		GPU_DEVICE void enumerate(Thread g, RandomIt first, RandomIt last, Function f)
		{
			enumerate(g, first, distance(first, last), f);
		}

		template <class Thread, class RandomIt, class Size, class Function>
		GPU_DEVICE void enumerate(Thread g, RandomIt first, Size n, Function f)
		{
			offset_t thid = g.thread_rank();

			while (thid < n)
			{
				f(*(first + thid), thid);
				thid += g.size();
			}
		}
	}
}
