#include <gstl/algorithms/transform.cuh>

namespace gpu
{
	template <class RandomIt1, class RandomIt2, class UnaryOperation>
	GPU_DEVICE RandomIt2 transform(block_t g, RandomIt1 first, RandomIt1 last, RandomIt2 d_first, UnaryOperation unary_op)
	{
		diff_t len = distance(first, last);
		diff_t thid = g.thread_rank();
		diff_t offset = 0;

		while (offset + thid < len)
		{
			*(d_first + offset + thid) = unary_op(*(first + offset + thid));
			offset += g.size();
		}
		
		return d_first + len;
	}

	template <class RandomIt1, class RandomIt2, class UnaryOperation, int tile_size>
	GPU_DEVICE RandomIt2 transform(block_tile_t<tile_size> g, RandomIt1 first, RandomIt1 last, RandomIt2 d_first, UnaryOperation unary_op)
	{
		diff_t len = distance(first, last);
		diff_t thid = g.thread_rank();
		diff_t offset = 0;

		while (offset + thid < len)
		{
			*(d_first + offset + thid) = unary_op(*(first + offset + thid));
			offset += g.size();
		}

		return d_first + len;
	}

	template <class ForwardIt, class OutputIt, class UnaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform(ForwardIt first, ForwardIt last, OutputIt d_first, UnaryOperation unary_op)
	{
		while (first1 != last1)
			*d_first++ = unary_op(*first1++);

		return d_first;
	}

	template <class RandomIt1, class RandomIt2, class RandomIt3, class BinaryOperation>
	GPU_DEVICE RandomIt3 transform(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt3 d_first, BinaryOperation binary_op)
	{
		diff_t len = distance(first1, last1);
		diff_t thid = g.thread_rank();
		diff_t offset = 0;

		while (offset + thid < len)
		{
			*(d_first + offset + thid) = binary_op(*(first1 + offset + thid), *(first2 + offset + thid));
			offset += g.size();
		}

		return d_first + len;
	}

	template <class RandomIt1, class RandomIt2, class RandomIt3, class BinaryOperation, int tile_size>
	GPU_DEVICE RandomIt3 transform(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt3 d_first, BinaryOperation binary_op)
	{
		diff_t len = distance(first1, last1);
		diff_t thid = g.thread_rank();
		diff_t offset = 0;

		while (offset + thid < len)
		{
			*(d_first + offset + thid) = binary_op(*(first1 + offset + thid), *(first2 + offset + thid));
			offset += g.size();
		}

		return d_first + len;
	}

	template <class ForwardIt1, class ForwardIt2, class OutputIt, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, OutputIt d_first, BinaryOperation binary_op)
	{
		while (first1 != last1)
			*d_first++ = binary_op(*first1++, *first2++);

		return d_first;
	}
}