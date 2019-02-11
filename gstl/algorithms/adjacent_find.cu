#include <gstl/algorithms/adjacent_find.cuh>

#include <gstl/functional/function_object.cuh>
#include <gstl/utility/ballot.cuh>

namespace gpu
{
	template <class ForwardIt>
	GPU_DEVICE ForwardIt adjacent_find(block_t g, ForwardIt first, ForwardIt last)
	{
		return adjacent_find(g, first, last, equal_to<>());
	}

	template <class BlockTile, class ForwardIt>
	GPU_DEVICE ForwardIt adjacent_find(BlockTile g, ForwardIt first, ForwardIt last)
	{
		return adjacent_find(g, first, last, equal_to<>());
	}

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt adjacent_find(ForwardIt first, ForwardIt last)
	{
		return adjacent_find(first, last, equal_to<>());
	}

	template <class ForwardIt, class BinaryPredicate>
	GPU_DEVICE ForwardIt adjacent_find(block_t g, ForwardIt first, ForwardIt last, BinaryPredicate p)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		do
		{
			bool result = false;
			if (offset + thid + 1 < len)
				result = p(*(first + offset + thid), *(first + offset + thid + 1));

			offset_t index = first_index(g, result);
			if (index != g.size())
				return first + offset + index;

			offset += g.size();
		} while (offset + g.size() < len);

		return last;
	}

	template <class BlockTile, class ForwardIt, class BinaryPredicate>
	GPU_DEVICE ForwardIt adjacent_find(BlockTile g, ForwardIt first, ForwardIt last, BinaryPredicate p)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		do
		{
			bool result = false;
			if (offset + thid + 1 < len)
				result = p(*(first + offset + thid), *(first + offset + thid + 1));

			offset_t index = first_index(g, result);
			if (index != g.size())
				return first + offset + index;

			offset += g.size();
		} while (offset + g.size() < len);

		return last;
	}

	template <class ForwardIt, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt adjacent_find(ForwardIt first, ForwardIt last, BinaryPredicate p)
	{
		if (first == last)
			return last;

		ForwardIt next = first;
		++next;
		for (; next != last; ++next, ++first)
			if (p(*first, *next))
				return first;

		return last;
	}
}
