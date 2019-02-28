#include <gstl/algorithms/sort/odd_even_merge_sort.cuh>

#include <gstl/algorithms/range.cuh>
#include <gstl/functional/function_object.cuh>
#include <gstl/math/bit.cuh>
#include <gstl/utility/swap.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class Compare>
		GPU_DEVICE void odd_even_merge_sort(Thread g, RandomIt first, RandomIt last, Compare comp, arbitrary_tag tag)
		{
			int len = distance(first, last);
			auto maximal_block_size = floor2(len);

			for (auto block_size = maximal_block_size; block_size > 0; block_size >>= 1)
			{
				range(g, len - block_size, [&](offset_t index) {
					if (!(index & block_size))
					{
						if (!comp(*(first + index), *(first + index + block_size)))
							iter_swap(first + index, first + index + block_size);
					}
				});
				g.sync();

				for (auto stride_length = maximal_block_size; stride_length > block_size; stride_length >>= 1)
				{
					range(g, len - stride_length, [&](offset_t index) {
						if (!(index & block_size))
						{
							if (!comp(*(first + index + block_size), *(first + index + stride_length)))
								iter_swap(first + index + block_size, first + index + stride_length);
						}
					});
					g.sync();
				}
			}
		}
	}

	template <class RandomIt>
	GPU_DEVICE void odd_even_merge_sort(block_t g, RandomIt first, RandomIt last)
	{
		return odd_even_merge_sort(g, first, last, less<>());
	}

	template <class RandomIt, unsigned int tile_sz>
	GPU_DEVICE void odd_even_merge_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last)
	{
		return odd_even_merge_sort(g, first, last, less<>());
	}

	template <class RandomIt, class Compare>
	GPU_DEVICE void odd_even_merge_sort(block_t g, RandomIt first, RandomIt last, Compare comp)
	{
		return odd_even_merge_sort(g, first, last, comp, arbitrary_tag{});
	}

	template <class RandomIt, class Compare, unsigned int tile_sz>
	GPU_DEVICE void odd_even_merge_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp)
	{
		return odd_even_merge_sort(g, first, last, comp, arbitrary_tag{});
	}

	template <class RandomIt, class Compare, class SortTag>
	GPU_DEVICE void odd_even_merge_sort(block_t g, RandomIt first, RandomIt last, Compare comp, SortTag tag)
	{
		return detail::odd_even_merge_sort(g, first, last, comp, tag);
	}

	template <class RandomIt, class Compare, class SortTag, unsigned int tile_sz>
	GPU_DEVICE void odd_even_merge_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp, SortTag tag)
	{
		return detail::odd_even_merge_sort(g, first, last, comp, tag);
	}
}
