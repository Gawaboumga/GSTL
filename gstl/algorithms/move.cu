#include <gstl/algorithms/fill.cuh>

#include <gstl/numeric/exclusive_scan.cuh>

namespace gpu
{
	template <class RandomIt, class ForwardIt>
	GPU_DEVICE ForwardIt move(block_t g, RandomIt first, RandomIt last, ForwardIt d_first)
	{
		offset_t thid = g.thread_rank();
		auto len = distance(first, last);

		while (thid < len)
		{
			*(d_first + thid) = std::move(*(first + thid));
			thid += g.size();
		}

		return d_first + len;
	}

	template <class BlockTile, class RandomIt, class ForwardIt>
	GPU_DEVICE ForwardIt move(BlockTile g, RandomIt first, RandomIt last, ForwardIt d_first)
	{
		offset_t thid = g.thread_rank();
		auto len = distance(first, last);

		while (thid < len)
		{
			*(d_first + thid) = std::move(*(first + thid));
			thid += g.size();
		}

		return d_first + len;
	}

	template <class InputIt, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt move(InputIt first, InputIt last, OutputIt d_first)
	{
		while (first != last)
			*d_first++ = std::move(*first++);

		return d_first;
	}

	template <class RandomIt, class BidirIt>
	GPU_DEVICE BidirIt move_backward(block_t g, RandomIt first, RandomIt last, BidirIt d_last)
	{
	#if defined(GPU_DEBUG_ALGORITHM)
		ENSURE(d_last < first || d_last >= last);
	#endif // GPU_DEBUG_ALGORITHM

		offset_t thid = g.thread_rank();
		offset_t offset = 0;
		auto len = distance(first, last);

		while (offset < len)
		{
			// All threads of the block should sync at the same time
			typename std::iterator_traits<RandomIt>::value_type tmp;
			if (offset + thid < len)
				tmp = std::move(*(last - (offset + thid) - 1));
			g.sync();
			if (offset + thid < len)
				*(d_last - (offset + thid) - 1) = tmp;
			g.sync();

			offset += g.size();
		}

		return d_last - len;
	}

	template <class BlockTile, class RandomIt, class BidirIt>
	GPU_DEVICE BidirIt move_backward(BlockTile g, RandomIt first, RandomIt last, BidirIt d_last)
	{
	#if defined(GPU_DEBUG_ALGORITHM)
		ENSURE(d_last < first || d_last >= last);
	#endif // GPU_DEBUG_ALGORITHM

		offset_t thid = g.thread_rank();
		auto len = distance(first, last);

		while (thid < len)
		{
			// With warp, block tile takes care of syncing every threads within the tile
			auto tmp = std::move(*(last - thid - 1));
			g.sync();
			*(d_last - thid - 1) = tmp;
			g.sync();
			thid += g.size();
		}

		return d_last - len;
	}

	template <class BidirIt1, class BidirIt2>
	GPU_DEVICE GPU_CONSTEXPR BidirIt2 move_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last)
	{
	#if defined(GPU_DEBUG_ALGORITHM)
		ENSURE(d_last < first || d_last >= last);
	#endif // GPU_DEBUG_ALGORITHM

		while (first != last)
			*(--d_last) = move(*(--last));

		return d_last;
	}
}
