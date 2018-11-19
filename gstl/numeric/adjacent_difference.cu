#include <gstl/numeric/adjacent_difference.cuh>

#include <gstl/functional/function_object.cuh>

#include <gstl/utility/iterator.cuh>

namespace gpu
{
	template <class RandomInputIt, class RandomOutputIt>
	GPU_DEVICE RandomOutputIt adjacent_difference(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first)
	{
		return adjacent_difference(g, first, last, d_first, minus<>());
	}

	template <class BlockTile, class RandomInputIt, class RandomOutputIt>
	GPU_DEVICE RandomOutputIt adjacent_difference(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first)
	{
		return adjacent_difference(g, first, last, d_first, minus<>());
	}

	template <class InputIt, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt adjacent_difference(InputIt first, InputIt last, OutputIt d_first)
	{
		return adjacent_difference(first, last, d_first, minus<>());
	}

	template <class RandomInputIt, class RandomOutputIt, class BinaryOperation>
	GPU_DEVICE RandomOutputIt adjacent_difference(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation op)
	{
		offset_t len = distance(first, last);
		offset_t offset = 0;

		while (offset + g.thread_rank() < len)
		{
			if (offset == 0 && g.thread_rank() == 0)
				*d_first = *first;
			else
				*(d_first + offset + g.thread_rank()) = op(*(first + offset + g.thread_rank()), *(first + offset + g.thread_rank() - 1));

			offset += g.size();
		}
		return d_first + len;
	}

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, class BinaryOperation>
	GPU_DEVICE RandomOutputIt adjacent_difference(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation op)
	{
		offset_t len = distance(first, last);
		offset_t offset = 0;

		while (offset + g.thread_rank() + 1 < len)
		{
			if (offset == 0 && g.thread_rank() == 0)
				*d_first = *first;
			else
				*(d_first + offset + g.thread_rank()) = op(*(first + offset + g.thread_rank() - 1), *(first + offset + g.thread_rank()));

			offset += g.size();
		}
		return d_first + len;
	}

	template< class InputIt, class OutputIt, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt adjacent_difference(InputIt first, InputIt last, OutputIt d_first, BinaryOperation op)
	{
		if (first == last)
			return d_first;

		using value_t = typename std::iterator_traits<InputIt>::value_type;
		value_t acc = *first;
		*d_first = acc;
		while (++first != last)
		{
			value_t val = *first;
			*++d_first = op(val, std::move(acc)); // std::move since C++20
			acc = std::move(val);
		}
		return ++d_first;
	}
}
