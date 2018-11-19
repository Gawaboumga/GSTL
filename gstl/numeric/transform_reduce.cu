#include <gstl/numeric/transform_reduce.cuh>

#include <gstl/functional/function_object.cuh>
#include <gstl/numeric/reduce.cuh>

namespace gpu
{
	template <class RandomIt1, class RandomIt2, typename T>
	GPU_DEVICE group_result<T> transform_reduce(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T init)
	{
		return transform_reduce(g, first1, last1, first2, init, plus<>(), multiplies<>());
	}

	template <class BlockTile, class RandomIt1, class RandomIt2, typename T>
	GPU_DEVICE group_result<T> transform_reduce(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T init)
	{
		return transform_reduce(g, first1, last1, first2, init, plus<>(), multiplies<>());
	}

	template <class InputIt1, class InputIt2, typename T>
	GPU_DEVICE GPU_CONSTEXPR T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init)
	{
		return transform_reduce(first1, last1, first2, init, plus<>(), multiplies<>());
	}

	template <class RandomIt1, class RandomIt2, typename T, class BinaryOp1, class BinaryOp2>
	GPU_DEVICE group_result<T> transform_reduce(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2)
	{
		block_tile_t<32> tile32 = tiled_partition<32>(g);
		GPU_SHARED T shared_data[MAX_NUMBER_OF_WARPS_PER_BLOCK];

		offset_t len = distance(first1, last1);
		offset_t thread_index = g.thread_rank();

		if (len < g.size())
		{
			T thread_result;
			if (g.thread_rank() < len)
			{
				thread_result = binary_op2(*(first1 + thread_index), *(first2 + thread_index));

				T thread_reduced_result = reduce(tile32, thread_result, binary_op1);
				if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK == 0)
					shared_data[g.thread_rank() / MAX_NUMBER_OF_WARPS_PER_BLOCK] = thread_reduced_result;
			} 
			g.sync();

			offset_t number_of_actives_warps = (len + MAX_NUMBER_OF_WARPS_PER_BLOCK - 1) / MAX_NUMBER_OF_WARPS_PER_BLOCK;
			if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK < number_of_actives_warps)
				thread_result = shared_data[g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK];
			thread_result = reduce(tile32, thread_result, binary_op1, number_of_actives_warps - 1);
			return binary_op1(thread_result, init);
		}
		else
		{
			T thread_result = binary_op2(*(first1 + thread_index), *(first2 + thread_index));
			thread_index += g.size();
			while (thread_index < len)
			{
				thread_result = binary_op1(thread_result, binary_op2(*(first1 + thread_index), *(first2 + thread_index)));
				thread_index += g.size();
			}

			T thread_reduced_result = reduce(tile32, thread_result, binary_op1);
			if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK == 0)
				shared_data[g.thread_rank() / MAX_NUMBER_OF_WARPS_PER_BLOCK] = thread_reduced_result;
			g.sync();

			offset_t number_of_actives_warps = g.size() / MAX_NUMBER_OF_WARPS_PER_BLOCK;
			if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK < number_of_actives_warps)
				thread_result = shared_data[g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK];
			thread_result = reduce(tile32, thread_result, binary_op1, number_of_actives_warps - 1);
			return binary_op1(thread_result, init);
		}
	}

	template <class BlockTile, class RandomIt1, class RandomIt2, typename T, class BinaryOp1, class BinaryOp2>
	GPU_DEVICE group_result<T> transform_reduce(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2)
	{
		offset_t len = distance(first1, last1);
		offset_t thread_index = g.thread_rank();

		if (len < g.size())
		{
			T thread_value;
			if (g.thread_rank() < len)
				thread_value = binary_op2(*(first1 + g.thread_rank()), *(first2 + g.thread_rank()));
			return binary_op1(reduce(g, thread_value, plus<>(), len - 1), init);
		}

		T thread_result = binary_op2(*(first1 + thread_index), *(first2 + thread_index));
		thread_index += g.size();
		while (thread_index < len)
		{
			thread_result = binary_op1(thread_result, binary_op2(*(first1 + thread_index), *(first2 + thread_index)));
			thread_index += g.size();
		}

		thread_result = reduce(g, thread_result, binary_op1);
		return binary_op1(thread_result, init);
	}

	template <class InputIt1, class InputIt2, typename T, class BinaryOp1, class BinaryOp2>
	GPU_DEVICE GPU_CONSTEXPR T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2)
	{
		while (first1 != last1)
		{
			init = binary_op1(std::move(init), binary_op2(*first1, *first2));
			++first1;
			++first2;
		}
		return init;
	}

	template <class RandomIt, typename T, class BinaryOp, class UnaryOp>
	GPU_DEVICE group_result<T> transform_reduce(block_t g, RandomIt first, RandomIt last, T init, BinaryOp binary_op, UnaryOp unary_op)
	{
		block_tile_t<32> tile32 = tiled_partition<32>(g);
		GPU_SHARED T shared_data[MAX_NUMBER_OF_WARPS_PER_BLOCK];

		offset_t len = distance(first, last);
		offset_t thread_index = g.thread_rank();

		if (len < g.size())
		{
			T thread_result;
			if (g.thread_rank() < len)
			{
				thread_result = unary_op(*(first + thread_index));

				group_result<T> thread_reduced_result = reduce(tile32, thread_result, binary_op);
				if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK == 0)
					shared_data[g.thread_rank() / MAX_NUMBER_OF_WARPS_PER_BLOCK] = thread_reduced_result;
			}
			g.sync();

			offset_t number_of_actives_warps = (len + MAX_NUMBER_OF_WARPS_PER_BLOCK - 1) / MAX_NUMBER_OF_WARPS_PER_BLOCK;
			if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK < number_of_actives_warps)
				thread_result = shared_data[g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK];
			thread_result = reduce(tile32, thread_result, binary_op, number_of_actives_warps - 1);
			return binary_op(thread_result, init);
		}
		else
		{
			T thread_result = unary_op(*(first + thread_index));
			thread_index += g.size();
			while (thread_index < len)
			{
				thread_result = binary_op(thread_result, unary_op(*(first + thread_index)));
				thread_index += g.size();
			}

			T thread_reduced_result = reduce(tile32, thread_result, binary_op);
			if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK == 0)
				shared_data[g.thread_rank() / MAX_NUMBER_OF_WARPS_PER_BLOCK] = thread_reduced_result;
			g.sync();

			offset_t number_of_actives_warps = g.size() / MAX_NUMBER_OF_WARPS_PER_BLOCK;
			if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK < number_of_actives_warps)
				thread_result = shared_data[g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK];
			thread_result = reduce(tile32, thread_result, binary_op, number_of_actives_warps - 1);
			return binary_op(thread_result, init);
		}
	}

	template <class BlockTile, class RandomIt, typename T, class BinaryOp, class UnaryOp>
	GPU_DEVICE group_result<T> transform_reduce(BlockTile g, RandomIt first, RandomIt last, T init, BinaryOp binary_op, UnaryOp unary_op)
	{
		offset_t len = distance(first, last);
		offset_t thread_index = g.thread_rank();

		if (len < g.size())
		{
			T thread_value;
			if (g.thread_rank() < len)
				thread_value = unary_op(*(first + g.thread_rank()));
			return binary_op(reduce(g, thread_value, plus<>(), len - 1), init);
		}

		T thread_result = unary_op(*(first + thread_index));
		thread_index += g.size();
		while (thread_index < len)
		{
			thread_result = binary_op(thread_result, unary_op(*(first + thread_index)));
			thread_index += g.size();
		}

		thread_result = reduce(g, thread_result, binary_op);
		return binary_op(thread_result, init);
	}

	template <class InputIt, typename T, class BinaryOp, class UnaryOp>
	GPU_DEVICE GPU_CONSTEXPR T transform_reduce(InputIt first, InputIt last, T init, BinaryOp binary_op, UnaryOp unary_op)
	{
		while (first != last)
		{
			init = binary_op(std::move(init), unary_op(*first));
			++first;
		}
		return init;
	}
}
