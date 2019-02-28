#include <gstl/algorithms/sort/shell_sort.cuh>

#include <gstl/algorithms/range.cuh>
#include <gstl/functional/function_object.cuh>
#include <gstl/math/bit.cuh>
#include <gstl/utility/swap.cuh>

namespace gpu
{
	namespace detail
	{
		template <class RandomIt, class Compare, class Size>
		GPU_DEVICE void insertion_sort(RandomIt first, Compare comp, Size max_size, Size gap)
		{
			for (auto j = gap; j < max_size; j += gap)
			{
				auto tmp = *(first + j);
				auto i = j - gap;
				while (i >= 0 && !comp(*(first + i), tmp))
				{
					*(first + i + gap) = *(first + i);
					i -= gap;
				}
				*(first + i + gap) = tmp;
			}
		}

		template <class Thread, class RandomIt, class Compare>
		GPU_DEVICE void shell_sort(Thread g, RandomIt first, RandomIt last, Compare comp, arbitrary_tag tag)
		{
			auto len = distance(first, last);

			for (auto gap = len >> 1; gap > 0; gap >>= 1)
			{
				range(g, gap, [&](offset_t index) {
					insertion_sort(first + index, comp, len - index, gap);
				});
				g.sync();
			}
		}
	}

	template <class RandomIt>
	GPU_DEVICE void shell_sort(block_t g, RandomIt first, RandomIt last)
	{
		return shell_sort(g, first, last, less<>());
	}

	template <class RandomIt, unsigned int tile_sz>
	GPU_DEVICE void shell_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last)
	{
		return shell_sort(g, first, last, less<>());
	}

	template <class RandomIt, class Compare>
	GPU_DEVICE void shell_sort(block_t g, RandomIt first, RandomIt last, Compare comp)
	{
		return shell_sort(g, first, last, comp, arbitrary_tag{});
	}

	template <class RandomIt, class Compare, unsigned int tile_sz>
	GPU_DEVICE void shell_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp)
	{
		return shell_sort(g, first, last, comp, arbitrary_tag{});
	}

	template <class RandomIt, class Compare, class SortTag>
	GPU_DEVICE void shell_sort(block_t g, RandomIt first, RandomIt last, Compare comp, SortTag tag)
	{
		return detail::shell_sort(g, first, last, comp, tag);
	}

	template <class RandomIt, class Compare, class SortTag, unsigned int tile_sz>
	GPU_DEVICE void shell_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp, SortTag tag)
	{
		return detail::shell_sort(g, first, last, comp, tag);
	}
}
