#include <gstl/grid/algorithms/fill.cuh>

#include <gstl/algorithms/detail/fill.cuh>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE void fill(grid_t grid, RandomIt first, RandomIt last, const T& value)
	{
		detail::fill(grid, first, last, value);
	}
}
