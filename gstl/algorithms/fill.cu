#include <gstl/algorithms/fill.cuh>

#include <gstl/algorithms/generate.cuh>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE void fill(block_t g, RandomIt first, RandomIt last, const T& value)
	{
		generate(g, first, last, [&value]() {
			return value;
		});
	}

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE void fill(BlockTile g, RandomIt first, RandomIt last, const T& value)
	{
		generate(g, first, last, [&value]() {
			return value;
		});
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR void fill(ForwardIt first, ForwardIt last, const T& value)
	{
		for (; first != last; ++first)
			*first = value;
	}
}
