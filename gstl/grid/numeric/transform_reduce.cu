#include <gstl/grid/numeric/transform_reduce.cuh>

#include <gstl/numeric/reduce.cuh>
#include <gstl/utility/group_result.cuh>

namespace gpu
{
	template <class RandomIt1, class RandomIt2, class RandomOutputIt, typename T>
	GPU_DEVICE void transform_reduce(grid_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomOutputIt buffer, T init)
	{
		return transform_reduce(g, first1, last1, first2, buffer, init, plus<>(), multiplies<>());
	}

	template <class RandomIt1, class RandomIt2, class RandomOutputIt, typename T, class BinaryOp1, class BinaryOp2>
	GPU_DEVICE void transform_reduce(grid_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomOutputIt buffer, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2)
	{
		offset_t len = distance(first1, last1);
		offset_t thid = g.thread_rank();

		if (len < g.size())
		{
			T thread_result;
			if (g.thread_rank() < len)
			{
				if (g.thread_rank() == 0)
					thread_result = binary_op1(init, binary_op2(*(first1 + thid), *(first2 + thid)));
				else
					thread_result = binary_op2(*(first1 + thid), *(first2 + thid));
			}

			block_t block = this_thread_block();
			group_result<T> result;
			if (g.thread_rank() / block.size() < len / block.size())
				result = reduce(block, thread_result, binary_op1);
			else
				result = reduce(block, thread_result, binary_op1, g.thread_rank() % block.size());

			if (block.thread_rank() == 0)
				buffer[block.group_index().x] = result;
			return;
		}

		T thread_result;
		if (g.thread_rank() == 0)
			thread_result = binary_op1(init, binary_op2(*(first1 + thid), *(first2 + thid)));
		else
			thread_result = binary_op2(*(first1 + thid), *(first2 + thid));

		thid += g.size();
		while (thid < len)
		{
			thread_result = binary_op1(thread_result, binary_op2(*(first1 + thid), *(first2 + thid)));
			thid += g.size();
		}

		block_t block = this_thread_block();
		group_result<T> result = reduce(block, thread_result, binary_op1);
		if (block.thread_rank() == 0)
			buffer[block.group_index().x] = result;
	}

	template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp, class UnaryOp>
	GPU_DEVICE void transform_reduce(grid_t g, RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op, UnaryOp unary_op)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();

		if (len < g.size())
		{
			T thread_result;
			if (g.thread_rank() < len)
			{
				if (g.thread_rank() == 0)
					thread_result = binary_op(init, unary_op(*(first + thid)));
				else
					thread_result = unary_op(*(first + thid));
			}

			block_t block = this_thread_block();
			group_result<T> result;
			if (g.thread_rank() / block.size() < len / block.size())
				result = reduce(block, thread_result, binary_op);
			else
				result = reduce(block, thread_result, binary_op, g.thread_rank() % block.size());

			if (block.thread_rank() == 0)
				buffer[block.group_index().x] = result;
			return;
		}

		T thread_result;
		if (g.thread_rank() == 0)
			thread_result = binary_op(init, unary_op(*(first + thid)));
		else
			thread_result = unary_op(*(first + thid));

		thid += g.size();
		while (thid < len)
		{
			thread_result = binary_op(thread_result, unary_op(*(first + thid)));
			thid += g.size();
		}

		block_t block = this_thread_block();
		group_result<T> result = reduce(block, thread_result, binary_op);
		if (block.thread_rank() == 0)
			buffer[block.group_index().x] = result;
	}
}
