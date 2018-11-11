#include <gstl/algorithms/fill.cuh>

#include <gstl/algorithms/for_each.cuh>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE void fill(block_t g, RandomIt first, RandomIt last, const T& value)
	{
		for_each(g, first, last, [&value](const T& v) {
			return value;
		});
	}

	template <class RandomIt, typename T, int tile_size>
	GPU_DEVICE void fill(block_tile_t<tile_size> g, RandomIt first, RandomIt last, const T& value)
	{
		for_each(g, first, last, [&value](const T& v) {
			return value;
		});
	}

	template <class ForwardIt, class T>
	GPU_DEVICE GPU_CONSTEXPR void fill(ForwardIt first, ForwardIt last, const T& value)
	{
		for (; first != last; ++first)
			*first = value;
	}
}
