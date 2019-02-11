#include <gstl/algorithms/detail/range.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class Size, class Function>
		GPU_DEVICE void range(Thread g, Size stop, Function f)
		{
			offset_t thid = g.thread_rank();

			while (thid < stop)
			{
				f(thid);
				thid += g.size();
			}
		}

		template <class Thread, class Size, class Function>
		GPU_DEVICE void range(Thread g, Size start, Size stop, Function f)
		{
			offset_t thid = g.thread_rank() + start;

			while (thid < stop)
			{
				f(thid);
				thid += g.size();
			}
		}

		template <class Thread, class Size, class Function>
		GPU_DEVICE void range(Thread g, Size start, Size stop, Size step, Function f)
		{
			offset_t thid = g.thread_rank() * step + start;

			while (thid < stop)
			{
				f(thid);
				thid += g.size() * step;
			}
		}
	}
}
