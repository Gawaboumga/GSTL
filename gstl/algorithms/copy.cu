#include <gstl/algorithms/fill.cuh>

#include <gstl/numeric/exclusive_scan.cuh>

namespace gpu
{
	template <class RandomIt, class ForwardIt>
	GPU_DEVICE ForwardIt copy(block_t g, RandomIt first, RandomIt last, ForwardIt d_first)
	{
		offset_t len = distance(first, last);
		return copy_n(g, first, len, d_first);
	}

	template <class BlockTile, class RandomIt, class ForwardIt>
	GPU_DEVICE ForwardIt copy(BlockTile g, RandomIt first, RandomIt last, ForwardIt d_first)
	{
		offset_t len = distance(first, last);
		return copy_n(g, first, len, d_first);
	}

	template <class InputIt, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt copy(InputIt first, InputIt last, OutputIt d_first)
	{
		while (first != last)
			*d_first++ = *first++;

		return d_first;
	}

	template <class RandomIt, class ForwardIt, class UnaryPredicate>
	GPU_DEVICE ForwardIt copy_if(block_t g, RandomIt first, RandomIt last, ForwardIt d_first, UnaryPredicate p)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;
		offset_t destination_offset = 0;

		do
		{
			offset_t relative_offset = 0;
			bool is_valid = false;
			if (offset + thid < len)
				is_valid = p(*(first + offset + thid));
			if (is_valid)
				relative_offset = 1;

			relative_offset = exclusive_scan(g, relative_offset, 0);

			if (is_valid)
				*(d_first + destination_offset + relative_offset) = *(first + offset + thid);
			offset += g.size();
			relative_offset = shfl(g, relative_offset, g.size() - 1);
			destination_offset += relative_offset;
		} while (offset < len);

		return d_first + destination_offset;
	}

	template <class BlockTile, class RandomIt, class ForwardIt, class UnaryPredicate>
	GPU_DEVICE ForwardIt copy_if(BlockTile g, RandomIt first, RandomIt last, ForwardIt d_first, UnaryPredicate p)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t destination_offset = 0;

		do
		{
			offset_t relative_offset = 0;
			bool is_valid = false;
			if (thid < len)
				is_valid = p(*(first + thid));
			if (is_valid)
				relative_offset = 1;

			relative_offset = exclusive_scan(g, relative_offset, 0);

			if (is_valid)
				*(d_first + destination_offset + relative_offset) = *(first + thid);
			thid += g.size();
			relative_offset = shfl(g, relative_offset, g.size() - 1);
			destination_offset += relative_offset;
		} while (thid < len);

		return d_first + destination_offset;
	}

	template <class InputIt, class OutputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR OutputIt copy_if(InputIt first, InputIt last, OutputIt d_first, UnaryPredicate p)
	{
		while (first != last)
		{
			if (pred(*first))
				*d_first++ = *first;
			first++;
		}

		return d_first;
	}

	template <class RandomIt, class Size, class ForwardIt>
	GPU_DEVICE ForwardIt copy_n(block_t g, RandomIt first, Size count, ForwardIt d_first)
	{
		offset_t thid = g.thread_rank();

		while (thid < count)
		{
			*(d_first + thid) = *(first + thid);
			thid += g.size();
		}
	}

	template <class BlockTile, class RandomIt, class Size, class ForwardIt>
	GPU_DEVICE ForwardIt copy_n(BlockTile g, RandomIt first, Size count, ForwardIt d_first)
	{
		offset_t thid = g.thread_rank();

		while (thid < count)
		{
			*(d_first + thid) = *(first + thid);
			thid += g.size();
		}
	}

	template <class InputIt, class Size, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt copy_n(InputIt first, Size count, OutputIt d_first)
	{
		if (count > 0)
		{
			*d_first++ = *first;
			for (Size i = 1; i < count; ++i)
				*d_first++ = *++first;
		}
		return d_first;
	}

	template <class BidirIt1, class BidirIt2>
	GPU_DEVICE GPU_CONSTEXPR BidirIt2 copy_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last)
	{
		while (first != last)
			*(--d_last) = *(--last);

		return d_last;
	}
}
