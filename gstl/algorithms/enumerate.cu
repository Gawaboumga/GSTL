#include <gstl/algorithms/enumerate.cuh>

namespace gpu
{
	template <class RandomIt, class Function>
	GPU_DEVICE void enumerate(block_t g, RandomIt first, RandomIt last, Function f)
	{
		offset_t len = distance(first, last);
		offset_t thid = block.thread_rank();
		offset_t offset = 0;

		while (offset + thid < len)
		{
			f(*(first + offset + thid), offset + thid);
			offset += block.size();
		}
	}

	template <class RandomIt, class Function, int tile_size>
	GPU_DEVICE void enumerate(block_tile_t<tile_size> g, RandomIt first, RandomIt last, Function f)
	{
		offset_t len = distance(first, last);
		offset_t thid = warp.thread_rank();
		offset_t offset = 0;

		while (offset + thid < len)
		{
			f(*(first + offset + thid), offset + thid);
			offset += warp.size();
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
