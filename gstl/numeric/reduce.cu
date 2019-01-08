#include <gstl/numeric/reduce.cuh>

#include <gstl/functional/function_object.cuh>

namespace gpu
{
	template <class RandomIt>
	GPU_DEVICE group_result<reduce_return_type<RandomIt>> reduce(block_t g, RandomIt first, RandomIt last)
	{
		return reduce(g, first, last, reduce_return_type<RandomIt>{});
	}

	template <class BlockTile, class RandomIt>
	GPU_DEVICE group_result<reduce_return_type<RandomIt>> reduce(BlockTile g, RandomIt first, RandomIt last)
	{
		return reduce(g, first, last, reduce_return_type<RandomIt>{});
	}

	template <class InputIt>
	GPU_DEVICE reduce_return_type<InputIt> reduce(InputIt first, InputIt last)
	{
		return reduce(first, last, reduce_return_type<InputIt>{});
	}

	template <class RandomIt, typename T>
	GPU_DEVICE group_result<T> reduce(block_t g, RandomIt first, RandomIt last, T init)
	{
		return reduce(g, first, last, init, plus<>());
	}

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE group_result<T> reduce(BlockTile g, RandomIt first, RandomIt last, T init)
	{
		return reduce(g, first, last, init, plus<>());
	}

	template <class InputIt, typename T>
	GPU_DEVICE T reduce(InputIt first, InputIt last, T init)
	{
		return reduce(first, last, init, plus<>());
	}

	template <class RandomIt, typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(block_t g, RandomIt first, RandomIt last, T init, BinaryOp binary_op)
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
				thread_result = *(first + thread_index);

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
			T thread_result = *(first + thread_index);
			thread_index += g.size();
			while (thread_index < len)
			{
				thread_result = binary_op(thread_result, *(first + thread_index));
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

	template <class BlockTile, class RandomIt, typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(BlockTile g, RandomIt first, RandomIt last, T init, BinaryOp binary_op)
	{
		offset_t len = distance(first, last);
		offset_t thread_index = g.thread_rank();

		if (len < g.size())
		{
			T thread_value;
			if (g.thread_rank() < len)
				thread_value = *(first + g.thread_rank());
			return binary_op(reduce(g, thread_value, plus<>(), len - 1), init);
		}

		T thread_result = *(first + thread_index);
		thread_index += g.size();
		while (thread_index < len)
		{
			thread_result = binary_op(thread_result, *(first + thread_index));
			thread_index += g.size();
		}

		thread_result = reduce(g, thread_result, binary_op);
		return binary_op(thread_result, init);
	}

	template <class InputIt, typename T, class BinaryOp>
	GPU_DEVICE T reduce(InputIt first, InputIt last, T init, BinaryOp binary_op)
	{
		for (; first != last; ++first)
		{
			init = op(std::move(init), *first);
		}
		return init;
	}

	template <typename T>
	GPU_DEVICE group_result<T> reduce(block_t g, T value)
	{
		return reduce(g, value, plus<>());
	}

	template <class BlockTile, typename T>
	GPU_DEVICE group_result<T> reduce(BlockTile g, T value)
	{
		return reduce(g, value, plus<>());
	}

	template <class BlockTile, typename T>
	GPU_DEVICE group_result<T> reduce(BlockTile g, T value, unsigned int maximal_lane)
	{
		return reduce(g, value, plus<>(), maximal_lane);
	}

	template <typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(block_t g, T value, BinaryOp binary_op)
	{
		block_tile_t<32> warp = tiled_partition<32>(g);
		GPU_SHARED T shared_data[MAX_NUMBER_OF_WARPS_PER_BLOCK];

		group_result<T> result = reduce(warp, value, binary_op);
		if (warp.thread_rank() == 0)
			shared_data[g.thread_rank() / 32] = result;
		g.sync();

		offset_t number_of_actives_warps = g.size() / MAX_NUMBER_OF_WARPS_PER_BLOCK;
		T thread_result;
		if (g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK < number_of_actives_warps)
			thread_result = shared_data[g.thread_rank() % MAX_NUMBER_OF_WARPS_PER_BLOCK];
		warp.sync();
		return reduce(warp, thread_result, binary_op, number_of_actives_warps - 1);
	}

	template <class BlockTile, typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(BlockTile g, T value, BinaryOp binary_op)
	{
		for (unsigned int i = g.size() / 2; i > 0; i /= 2)
			value = binary_op(value, g.shfl_down(value, i));

		return value;
	}

	template <class BlockTile, typename T, class BinaryOp>
	GPU_DEVICE group_result<T> reduce(BlockTile g, T value, BinaryOp binary_op, unsigned int maximal_lane)
	{
		if (g.thread_rank() <= maximal_lane)
		{
			unsigned mask = __activemask();
			for (unsigned int i = g.size() / 2; i > 0; i /= 2)
				value = binary_op(value, __shfl_down_sync(mask, value, i));
		}
		value = shfl(g, value);
		return value;
	}
}
