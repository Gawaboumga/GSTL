#include <gstl/algorithms/swap.cuh>

namespace gpu
{
	template <class ForwardIt1, class ForwardIt2>
	GPU_DEVICE ForwardIt2 swap_ranges(block_t g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2)
	{
		auto len = distance(first1, last1);
		offset_t thid = g.thread_rank();

		while (thid < len)
		{
			using gpu::iter_swap;
			iter_swap(first1 + thid, first2 + thid);
			thid += g.size();
		}
	}

	template <class BlockTile, class ForwardIt1, class RandomIt2>
	GPU_DEVICE ForwardIt2 swap_ranges(BlockTile g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2)
	{
		auto len = distance(first1, last1);
		offset_t thid = g.thread_rank();

		while (thid < len)
		{
			using gpu::iter_swap;
			iter_swap(first1 + thid, first2 + thid);
			thid += g.size();
		}
	}

	template <class ForwardIt1, class ForwardIt2>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt2 swap_ranges(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2)
	{
		using gpu::iter_swap;
		while (first1 != last1)
			iter_swap(first1++, first2++);

		return first2;
	}
}
