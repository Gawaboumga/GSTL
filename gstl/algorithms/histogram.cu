#include <gstl/algorithms/histogram.cuh>

#include <gstl/algorithms/for_each.cuh>

namespace gpu
{
	namespace detail
	{
		struct incrementer
		{
			template <typename T>
			GPU_DEVICE GPU_CONSTEXPR auto operator()(T&& arg) const -> decltype(++static_cast<T&&>(arg))
			{
				return (++arg);
			}
		};
	}

	template <class RandomIt, class OutputIt, class MappingFunction>
	GPU_DEVICE void histogram(block_t g, RandomIt first, RandomIt last, OutputIt d_first, MappingFunction unary_op)
	{
		return histogram(g, first, last, d_first, unary_op, detail::incrementer{});
	}

	template <class BlockTile, class RandomIt, class OutputIt, class MappingFunction>
	GPU_DEVICE void histogram(BlockTile g, RandomIt first, RandomIt last, OutputIt d_first, MappingFunction unary_op)
	{
		return histogram(g, first, last, d_first, unary_op, detail::incrementer{});
	}

	template <class Input, class OutputIt, class MappingFunction>
	GPU_DEVICE GPU_CONSTEXPR void histogram(Input first, Input last, OutputIt d_first, MappingFunction unary_op)
	{
		return histogram(first, last, d_first, unary_op, detail::incrementer{});
	}

	template <class RandomIt, class OutputIt, class MappingFunction, class Incrementer>
	GPU_DEVICE void histogram(block_t g, RandomIt first, RandomIt last, OutputIt d_first, MappingFunction unary_op, Incrementer f)
	{
		for_each(g, first, last, [&d_first, &unary_op, &f](auto& value) {
			f(*(d_first + unary_op(value)));
		});
	}

	template <class BlockTile, class RandomIt, class OutputIt, class MappingFunction, class Incrementer>
	GPU_DEVICE void histogram(BlockTile g, RandomIt first, RandomIt last, OutputIt d_first, MappingFunction unary_op, Incrementer f)
	{
		for_each(g, first, last, [&d_first, &unary_op, &f](auto& value) {
			f(*(d_first + unary_op(value)));
		});
	}

	template <class Input, class OutputIt, class MappingFunction, class Incrementer>
	GPU_DEVICE GPU_CONSTEXPR void histogram(Input first, Input last, OutputIt d_first, MappingFunction unary_op, Incrementer f)
	{
		for (; first != last; ++first)
			f(*(d_first + unary_op(*first)));
	}
}
