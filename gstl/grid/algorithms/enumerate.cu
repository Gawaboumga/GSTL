#include <gstl/grid/algorithms/enumerate.cuh>

#include <gstl/algorithms/detail/enumerate.cuh>

namespace gpu
{
	template <class RandomIt, class Function>
	GPU_DEVICE void enumerate(grid_t grid, RandomIt first, RandomIt last, Function f)
	{
		detail::enumerate(grid, first, last, f);
	}

	template <class RandomIt, class Size, class Function>
	GPU_DEVICE void enumerate(grid_t grid, RandomIt first, Size n, Function f)
	{
		detail::enumerate(grid, first, n, f);
	}
}
