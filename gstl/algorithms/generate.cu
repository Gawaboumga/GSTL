#include <gstl/algorithms/generate.cuh>

namespace gpu
{
	template <class RandomIt, class Generator>
	GPU_DEVICE void generate(block_t block, RandomIt first, RandomIt last, Generator g)
	{
		offset_t len = distance(first, last);
		offset_t thid = block.thread_rank();
		offset_t offset = 0;

		while (offset + thid < len)
		{
			*(first + offset + thid) = g();
			offset += block.size();
		}
	}

	template <class RandomIt, class Generator, int tile_size>
	GPU_DEVICE void generate(block_tile_t<tile_size> warp, RandomIt first, RandomIt last, Generator g)
	{
		offset_t len = distance(first, last);
		offset_t thid = warp.thread_rank();
		offset_t offset = 0;

		while (offset + thid < len)
		{
			*(first + offset + thid) = g();
			offset += warp.size();
		}
	}

	template <class ForwardIt, class Generator>
	GPU_DEVICE GPU_CONSTEXPR void generate(ForwardIt first, ForwardIt last, Generator g)
	{
		while (first != last)
			*first++ = g();
	}
}
