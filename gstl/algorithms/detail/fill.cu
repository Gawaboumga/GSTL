#include <gstl/algorithms/detail/fill.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, typename T>
		GPU_DEVICE void fill(Thread g, RandomIt first, RandomIt last, const T& value)
		{
			return fill_n(g, first, distance(first, last), value);
		}

		template <class Thread, class ForwardIt, class Size, typename T>
		GPU_DEVICE void fill_n(Thread g, ForwardIt first, Size n, const T& value)
		{
			offset_t thid = g.thread_rank();

			while (thid < n)
			{
				*(first + thid) = value;
				thid += g.size();
			}
		}
	}
}
