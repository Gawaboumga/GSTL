#include <gstl/algorithms/histogram.cuh>

#include <gstl/algorithms/for_each.cuh>

namespace gpu
{
	template <class RandomIt, class OutputIt, class MappingFunction>
	GPU_DEVICE void histogram(block_t g, RandomIt first, RandomIt last, OutputIt d_first, MappingFunction unary_op)
	{
		for_each(g, first, last, [&d_first, &unary_op](auto& value) {
			*(d_first + unary_op(value)) += 1;
		});
	}

	template <class BlockTile, class RandomIt, class OutputIt, class MappingFunction>
	GPU_DEVICE void histogram(BlockTile g, RandomIt first, RandomIt last, OutputIt d_first, MappingFunction unary_op)
	{
		for_each(g, first, last, [&d_first, &unary_op](auto& value) {
			*(d_first + unary_op(value)) += 1;
		});
	}

	template <class Input, class OutputIt, class MappingFunction>
	GPU_DEVICE GPU_CONSTEXPR void histogram(Input first, Input last, OutputIt d_first, MappingFunction unary_op)
	{
		for (; first != last; ++first)
			*(d_first + unary_op(*first)) += 1;
	}
}
