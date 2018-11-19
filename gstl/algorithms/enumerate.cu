#include <gstl/algorithms/enumerate.cuh>

namespace gpu
{
	template <class RandomIt, class Function>
	GPU_DEVICE void enumerate(block_t g, RandomIt first, RandomIt last, Function f)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		while (offset + thid < len)
		{
			f(*(first + offset + thid), offset + thid);
			offset += g.size();
		}
	}

	template <class BlockTile, class RandomIt, class Function>
	GPU_DEVICE void enumerate(BlockTile g, RandomIt first, RandomIt last, Function f)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		while (offset + thid < len)
		{
			f(*(first + offset + thid), offset + thid);
			offset += g.size();
		}
	}

	template <class ForwardIt, class Function>
	GPU_DEVICE GPU_CONSTEXPR void enumerate(ForwardIt first, ForwardIt last, Function f)
	{
		offset_t offset = 0;
		while (first != last)
		{
			f(*first++, offset);
			++offset;
		}
	}
}