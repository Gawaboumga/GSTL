#include <gstl/algorithms/detail/generate.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class Generator>
		GPU_DEVICE void generate(Thread g, RandomIt first, RandomIt last, Generator gen)
		{
			return generate_n(g, first, distance(first, last), gen);
		}

		template <class Thread, class ForwardIt, class Size, class Generator>
		GPU_DEVICE void generate_n(Thread g, ForwardIt first, Size n, Generator gen)
		{
			offset_t thid = g.thread_rank();

			while (thid < n)
			{
				*(first + thid) = gen();
				thid += g.size();
			}
		}
	}
}
