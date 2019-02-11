#include <gstl/algorithms/range.cuh>

#include <gstl/algorithms/detail/range.cuh>

namespace gpu
{
	template <class Thread, class Size, class Function>
	GPU_DEVICE void range(Thread g, Size stop, Function f)
	{
		detail::range(g, stop, f);
	}

	template <class Thread, class Size, class Function>
	GPU_DEVICE void range(Thread g, Size start, Size stop, Function f)
	{
		detail::range(g, start, stop, f);
	}

	template <class Thread, class Size, class Function>
	GPU_DEVICE void range(Thread g, Size start, Size stop, Size step, Function f)
	{
		detail::range(g, start, stop, step, f);
	}
}
