#include <gstl/algorithms/sort/bitonic_sort.cuh>

#include <gstl/algorithms/range.cuh>
#include <gstl/functional/function_object.cuh>
#include <gstl/math/bit.cuh>
#include <gstl/utility/swap.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class Compare>
		GPU_DEVICE void bitonic_sort(Thread g, RandomIt first, RandomIt last, Compare comp, arbitrary_tag tag)
		{
			auto len = distance(first, last);
			for (offset_t block_size = 2u; block_size <= len * 2; block_size <<= 1)
			{
				offset_t mask_block_size = block_size - 1;

				range(g, len - (block_size >> 2), [&](offset_t index) {
					offset_t local_index = index / (block_size / 2);
					offset_t local_offset = index % (block_size / 2);
					offset_t local_start = local_index * block_size;

					offset_t i = local_start + local_offset;
					offset_t j = local_start + block_size - local_offset - 1;

					if (j >= len)
						return;

					if (!comp(*(first + i), *(first + j)))
						iter_swap(first + i, first + j);
				});
				g.sync();

				for (offset_t stride_length = block_size >> 1; stride_length > 0; stride_length >>= 1)
				{
					offset_t mask_stride_length = stride_length - 1;

					range(g, len >> 1, [&](offset_t index) {
						offset_t local_index = (index << 1) - (index & mask_stride_length);

						offset_t i = (local_index << 1) - (local_index & mask_block_size);
						offset_t j = i ^ stride_length;

						if (j >= len)
							return;

						if (!comp(*(first + i), *(first + j)))
							iter_swap(first + i, first + j);

						i |= block_size;
						j = i ^ stride_length;

						if (j >= len)
							return;

						if (!comp(*(first + i), *(first + j)))
							iter_swap(first + i, first + j);
					});
					g.sync();
				}
			}
		}

		template <class Thread, class RandomIt, class Compare>
		GPU_DEVICE void bitonic_sort(Thread g, RandomIt first, RandomIt last, Compare comp, power_of_two_tag tag)
		{
			auto len = distance(first, last);
			for (offset_t block_size = 2u; block_size < len; block_size <<= 1)
			{
				offset_t mask_block_size = block_size - 1;

				for (offset_t stride_length = block_size >> 1; stride_length > 0; stride_length >>= 1)
				{
					offset_t mask_stride_length = stride_length - 1;

					range(g, len >> 2, [&](offset_t index) {
						offset_t local_index = (index << 1) - (index & mask_stride_length);

						offset_t i = (local_index << 1) - (local_index & mask_block_size);
						offset_t j = i ^ stride_length;

						if (comp(*(first + i), *(first + j)))
							iter_swap(first + i, first + j);

						i |= block_size;
						j = i ^ stride_length;

						if (!comp(*(first + i), *(first + j)))
							iter_swap(first + i, first + j);
					});
					g.sync();
				}
			}

			for (offset_t stride_length = len >> 1; stride_length > 0; stride_length >>= 1)
			{
				offset_t mask_stride_length = stride_length - 1;

				range(g, len >> 1, [&](offset_t index) {
					offset_t i = (index << 1) - (index & mask_stride_length);
					offset_t j = i ^ stride_length;

					if (!comp(*(first + i), *(first + j)))
						iter_swap(first + i, first + j);
				});
				g.sync();
			}
		}
	}

	template <class RandomIt>
	GPU_DEVICE void bitonic_sort(block_t g, RandomIt first, RandomIt last)
	{
		return bitonic_sort(g, first, last, less<>());
	}

	template <class RandomIt, unsigned int tile_sz>
	GPU_DEVICE void bitonic_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last)
	{
		return bitonic_sort(g, first, last, less<>());
	}

	template <class RandomIt, class Compare>
	GPU_DEVICE void bitonic_sort(block_t g, RandomIt first, RandomIt last, Compare comp)
	{
		auto len = distance(first, last);
		if (ispow2(len))
			return bitonic_sort(g, first, last, comp, power_of_two_tag{});
		else
			return bitonic_sort(g, first, last, comp, arbitrary_tag{});
	}

	template <class RandomIt, class Compare, unsigned int tile_sz>
	GPU_DEVICE void bitonic_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp)
	{
		auto len = distance(first, last);
		if (ispow2(len))
			return bitonic_sort(g, first, last, comp, power_of_two_tag{});
		else
			return bitonic_sort(g, first, last, comp, arbitrary_tag{});
	}

	template <class RandomIt, class Compare, class SortTag>
	GPU_DEVICE void bitonic_sort(block_t g, RandomIt first, RandomIt last, Compare comp, SortTag tag)
	{
		return detail::bitonic_sort(g, first, last, comp, tag);
	}

	template <class RandomIt, class Compare, class SortTag, unsigned int tile_sz>
	GPU_DEVICE void bitonic_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp, SortTag tag)
	{
		return detail::bitonic_sort(g, first, last, comp, tag);
	}
}
