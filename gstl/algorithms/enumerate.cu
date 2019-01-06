#include <gstl/algorithms/enumerate.cuh>

#include <gstl/algorithms/detail/for_each.cuh>

namespace gpu
{
	template <class RandomIt, class Function>
	GPU_DEVICE void enumerate(block_t g, RandomIt first, RandomIt last, Function f)
	{
		detail::enumerate(g, first, last, f);
	}

	template <class BlockTile, class RandomIt, class Function>
	GPU_DEVICE void enumerate(BlockTile g, RandomIt first, RandomIt last, Function f)
	{
		detail::enumerate(g, first, last, f);
	}

	template <class ForwardIt, class Function>
	GPU_DEVICE GPU_CONSTEXPR void enumerate(ForwardIt first, ForwardIt last, Function f)
	{
		offset_t offset = 0;
		while (first != last)
		{
			f(*first++, offset);
			++offset;
		}
	}
}
