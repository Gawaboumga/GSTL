#include <gstl/algorithms/generate.cuh>

#include <gstl/algorithms/detail/generate.cuh>

namespace gpu
{
	template <class RandomIt, class Generator>
	GPU_DEVICE void generate(block_t g, RandomIt first, RandomIt last, Generator gen)
	{
		detail::generate(g, first, last, gen);
	}

	template <class BlockTile, class RandomIt, class Generator>
	GPU_DEVICE void generate(BlockTile g, RandomIt first, RandomIt last, Generator gen)
	{
		detail::generate(g, first, last, gen);
	}

	template <class ForwardIt, class Generator>
	GPU_DEVICE GPU_CONSTEXPR void generate(ForwardIt first, ForwardIt last, Generator g)
	{
		while (first != last)
			*first++ = g();
	}
}
