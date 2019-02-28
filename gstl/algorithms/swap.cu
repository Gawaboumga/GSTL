#include <gstl/algorithms/swap.cuh>

#include <gstl/utility/swap.cuh>

namespace gpu
{
	template <class RandomIt1, class RandomIt2>
	GPU_DEVICE RandomIt2 swap_ranges(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2)
	{
		auto len = distance(first1, last1);
		offset_t thid = g.thread_rank();

		while (thid < len)
		{
			using gpu::iter_swap;
			iter_swap(first1 + thid, first2 + thid);
			thid += g.size();
		}

		return first2 + len;
	}

	template <class BlockTile, class RandomIt1, class RandomIt2>
	GPU_DEVICE RandomIt2 swap_ranges(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2)
	{
		auto len = distance(first1, last1);
		offset_t thid = g.thread_rank();

		while (thid < len)
		{
			using gpu::iter_swap;
			iter_swap(first1 + thid, first2 + thid);
			thid += g.size();
		}

		return first2 + len;
	}

	template <class ForwardIt1, class ForwardIt2>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt2 swap_ranges(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2)
	{
		using gpu::iter_swap;
		while (first1 != last1)
			iter_swap(first1++, first2++);

		return first2;
	}

	template <class RandomIt1, class RandomIt2, class Size>
	GPU_DEVICE RandomIt2 swap_ranges_n(block_t g, RandomIt1 first1, RandomIt2 first2, Size n)
	{
		offset_t thid = g.thread_rank();

		while (thid < n)
		{
			using gpu::iter_swap;
			iter_swap(first1 + thid, first2 + thid);
			thid += g.size();
		}

		return first2 + n;
	}

	template <class BlockTile, class RandomIt1, class RandomIt2, class Size>
	GPU_DEVICE RandomIt2 swap_ranges_n(BlockTile g, RandomIt1 first1, RandomIt2 first2, Size n)
	{
		offset_t thid = g.thread_rank();

		while (thid < n)
		{
			using gpu::iter_swap;
			iter_swap(first1 + thid, first2 + thid);
			thid += g.size();
		}

		return first2 + n;
	}

	template <class ForwardIt1, class ForwardIt2, class Size>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt2 swap_ranges_n(ForwardIt1 first1, ForwardIt2 first2, Size n)
	{
		using gpu::iter_swap;
		for (Size i = 0; i != n; ++i)
			iter_swap(first1++, first2++);

		return first2;
	}
}
