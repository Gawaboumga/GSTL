#include <gstl/algorithms/sort.cuh>

#include <gstl/algorithms/inplace_merge.cuh>
#include <gstl/algorithms/sort/odd_even_sort.cuh>
#include <gstl/functional/function_object.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class Compare>
		GPU_DEVICE void sort(Thread g, RandomIt first, RandomIt last, Compare comp)
		{
			constexpr offset_t BOTTOM_SIZE = 1 << 7;
			auto len = distance(first, last);

			for (offset_t i = 0; i < len; i += BOTTOM_SIZE)
			{
				if (i + BOTTOM_SIZE < len)
					odd_even_sort(g, first + i, first + i + BOTTOM_SIZE, comp);
				else
					odd_even_sort(g, first + i, last, comp);
			}

			offset_t merge_phase = BOTTOM_SIZE * 2;
			while (merge_phase < len * 2)
			{
				for (offset_t i = 0; i < len; i += merge_phase)
				{
					if (i + merge_phase < len)
						gpu::inplace_merge(g, first + i, first + i + merge_phase / 2, first + i + merge_phase, comp);
					else if (i + merge_phase / 2 < len)
						gpu::inplace_merge(g, first + i, first + i + merge_phase / 2, last, comp);
				}
				merge_phase *= 2;
			}
		}
	}

	template <class RandomIt>
	GPU_DEVICE void sort(block_t g, RandomIt first, RandomIt last)
	{
		return sort(g, first, last, less<>{});
	}

	template <class BlockTile, class RandomIt>
	GPU_DEVICE void sort(BlockTile g, RandomIt first, RandomIt last)
	{
		return sort(g, first, last, less<>{});
	}

	template <class RandomIt, class Compare>
	GPU_DEVICE void sort(block_t g, RandomIt first, RandomIt last, Compare comp)
	{
		detail::sort(g, first, last, comp);
	}

	template <class BlockTile, class RandomIt, class Compare>
	GPU_DEVICE void sort(BlockTile g, RandomIt first, RandomIt last, Compare comp)
	{
		detail::sort(g, first, last, comp);
	}
}
