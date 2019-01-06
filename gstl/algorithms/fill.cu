#include <gstl/algorithms/fill.cuh>

#include <gstl/algorithms/detail/fill.cuh>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE void fill(block_t g, RandomIt first, RandomIt last, const T& value)
	{
		detail::fill(g, first, last, value);
	}

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE void fill(BlockTile g, RandomIt first, RandomIt last, const T& value)
	{
		detail::fill(g, first, last, value);
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR void fill(ForwardIt first, ForwardIt last, const T& value)
	{
		for (; first != last; ++first)
			*first = value;
	}

	template <class RandomIt, class Size, typename T>
	GPU_DEVICE void fill_n(block_t g, RandomIt first, Size n, const T& value)
	{
		detail::fill_n(g, first, n, value);
	}

	template <class BlockTile, class RandomIt, class Size, typename T>
	GPU_DEVICE void fill_n(BlockTile g, RandomIt first, Size n, const T& value)
	{
		detail::fill_n(g, first, n, value);
	}

	template <class OutputIt, class Size, typename T>
	GPU_DEVICE GPU_CONSTEXPR OutputIt fill_n(OutputIt first, Size n, const T& value)
	{
		for (Size i = 0; i < count; i++)
			*first++ = value;

		return first;
	}
}
