#include <gstl/algorithms/sort/odd_even_sort.cuh>

#include <gstl/algorithms/range.cuh>
#include <gstl/functional/function_object.cuh>
#include <gstl/math/bit.cuh>
#include <gstl/utility/swap.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class Compare>
		GPU_DEVICE void odd_even_sort(Thread g, RandomIt first, RandomIt last, Compare comp, arbitrary_tag tag)
		{
			GPU_SHARED int exch0;
			GPU_SHARED int exch1;
			GPU_SHARED int trips;

			if (g.thread_rank() == 0)
			{
				exch1 = 1;
				trips = 0;
			}
			g.sync();

			int len = distance(first, last);
			while (exch1)
			{
				if (g.thread_rank() == 0)
				{
					exch0 = 0;
					exch1 = 0;
				}
				g.sync();

				range(g, 0, len - 1, 2, [&](offset_t index) {
					if (comp(*(first + index + 1), *(first + index)))
					{
						iter_swap(first + index, first + index + 1);
						exch0 = 1;
					}
				});

				if (exch0 || !trips)
				{
					range(g, 1, len - 1, 2, [&](offset_t index) {
						if (comp(*(first + index + 1), *(first + index)))
						{
							iter_swap(first + index, first + index + 1);
							exch1 = 1;
						}
					});
				}

				if (g.thread_rank() == 0)
					trips = 1;
				g.sync();
			}
		}
	}

	template <class RandomIt>
	GPU_DEVICE void odd_even_sort(block_t g, RandomIt first, RandomIt last)
	{
		return odd_even_sort(g, first, last, less<>());
	}

	template <class RandomIt, unsigned int tile_sz>
	GPU_DEVICE void odd_even_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last)
	{
		return odd_even_sort(g, first, last, less<>());
	}

	template <class RandomIt, class Compare>
	GPU_DEVICE void odd_even_sort(block_t g, RandomIt first, RandomIt last, Compare comp)
	{
		return odd_even_sort(g, first, last, comp, arbitrary_tag{});
	}

	template <class RandomIt, class Compare, unsigned int tile_sz>
	GPU_DEVICE void odd_even_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp)
	{
		return odd_even_sort(g, first, last, comp, arbitrary_tag{});
	}

	template <class RandomIt, class Compare, class SortTag>
	GPU_DEVICE void odd_even_sort(block_t g, RandomIt first, RandomIt last, Compare comp, SortTag tag)
	{
		return detail::odd_even_sort(g, first, last, comp, tag);
	}

	template <class RandomIt, class Compare, class SortTag, unsigned int tile_sz>
	GPU_DEVICE void odd_even_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp, SortTag tag)
	{
		return detail::odd_even_sort(g, first, last, comp, tag);
	}
}
