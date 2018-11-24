#include <gstl/numeric/transform_inclusive_scan.cuh>

#include <gstl/functional/function_object.cuh>
#include <gstl/utility/ballot.cuh>

#include <type_traits>

namespace gpu
{
	namespace detail
	{
		template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
		GPU_DEVICE void transform_inclusive_scan_partial(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
		{
			block_tile_t<32> warp = tiled_partition<32>(g);
			if (g.thread_rank() < warp.size())
			{
				transform_inclusive_scan(warp, first, last, d_first, binary_op, unary_op, init);
			}
			g.sync();
		}

		template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
		GPU_DEVICE RandomOutputIt  transform_inclusive_scan_full(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
		{
			offset_t n = 2 * g.size();
			offset_t thid = g.thread_rank();
			offset_t offset = 1;

			*(d_first + 2 * thid) = unary_op(*(first + 2 * thid));
			*(d_first + 2 * thid + 1) = unary_op(*(first + 2 * thid + 1));

			for (offset_t d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
			{
				g.sync();
				if (thid < d)
				{
					offset_t ai = offset * (2 * thid + 1) - 1;
					offset_t bi = offset * (2 * thid + 2) - 1;
					*(d_first + bi) = binary_op(*(d_first + bi), *(d_first + ai));
				}
				offset *= 2;
			}

			if (thid == 0)
				*(d_first + n - 1) = binary_op(*d_first, init); // clear the last element

			for (offset_t d = 1; d < n; d *= 2) // traverse down tree & build scan
			{
				offset >>= 1;
				g.sync();
				if (thid < d)
				{
					offset_t ai = offset * (2 * thid + 1) - 1;
					offset_t bi = offset * (2 * thid + 2) - 1;
					T tmp = *(d_first + ai);
					*(d_first + ai) = *(d_first + bi);
					*(d_first + bi) = binary_op(*(d_first + bi), tmp);
				}
			}
			g.sync();

			return d_first + n - 1;
		}
	}

	template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation>
	GPU_DEVICE RandomOutputIt transform_inclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op)
	{
		std::result_of_t<UnaryOperation(std::remove_reference_t<decltype(*first)>)> value{};
		return transform_inclusive_scan(g, first, last, d_first, binary_op, unary_op, value);
	}

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation>
	GPU_DEVICE RandomOutputIt transform_inclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op)
	{
		std::result_of_t<UnaryOperation(std::remove_reference_t<decltype(*first)>)> value{};
		return transform_inclusive_scan(g, first, last, d_first, binary_op, unary_op, value);
	}

	template <class InputIt, class OutputIt, class BinaryOperation, class UnaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform_inclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op)
	{
		std::result_of_t<UnaryOperation(std::remove_reference_t<decltype(*first)>)> value{};
		return transform_inclusive_scan(g, first, last, d_first, binary_op, unary_op, value);
	}

	template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
	GPU_DEVICE RandomOutputIt transform_inclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
	{
		offset_t len = distance(first, last);
		offset_t offset = 0;

		T value = init;
		if (len < 2 * g.size())
		{
			detail::transform_inclusive_scan_partial(g, first, last, d_first, binary_op, unary_op, value);
			return d_first + len;
		}

		auto result_ptr = detail::transform_inclusive_scan_full(g, first, last, d_first, binary_op, unary_op, value);
		value = shfl(g, *result_ptr, g.size() - 1);
		offset += 2 * g.size();

		while (offset < len && offset + 2 * g.size() < len)
		{
			auto result_ptr = detail::transform_inclusive_scan_full(g, first + offset, last, d_first + offset, binary_op, unary_op, value);
			value = shfl(g, *result_ptr, g.size() - 1);
			offset += 2 * g.size();
		}

		if (offset < len)
			detail::transform_inclusive_scan_partial(g, first + offset, last, d_first + offset, binary_op, unary_op, value);

		return d_first + len;
	}

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
	GPU_DEVICE RandomOutputIt transform_inclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
	{
		T current_value = init;
		offset_t len = distance(first, last);
		offset_t offset = 0;

		while (offset + g.thread_rank() < len)
		{
			T result = transform_inclusive_scan(g, unary_op(*(first + offset + g.thread_rank())), binary_op, current_value);
			*(d_first + offset + g.thread_rank()) = result;
			current_value = shfl(g, result, g.size() - 1);
			offset += g.size();
		}
		
		return d_first + len;
	}

	template <class InputIt, class OutputIt, class BinaryOperation, class UnaryOperation, typename T>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform_inclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
	{
		while (first != last)
		{
			*d_first = init;
			init = binary_op(init, *first);
			++first;
			++d_first;
		}
	}

	template <class BinaryOperation, typename T>
	GPU_DEVICE T transform_inclusive_scan(block_t g, T value, BinaryOperation binary_op, T init)
	{
		GPU_SHARED array<T, MAX_NUMBER_OF_WARPS_PER_BLOCK> warp_results;

		block_tile_t<MAX_NUMBER_OF_THREADS_PER_WARP> warp = tiled_partition<MAX_NUMBER_OF_THREADS_PER_WARP>(g);
		offset_t warp_id = g.thread_rank() / warp.size();
		T result = transform_inclusive_scan(warp, value, binary_op);

		if (warp.size() * (warp_id + 1) - 1 == g.thread_rank())
			warp_results[warp_id] = result;
		g.sync();

		if (g.thread_rank() < g.size() / warp.size())
		{
			T warp_scan = transform_inclusive_scan(warp, warp_results[warp_id], binary_op, 0);
			warp_results[g.thread_rank()] = warp_scan;
		}
		g.sync();

		if (warp_id != 0)
			result = binary_op(result, warp_results[warp_id - 1]);
		return binary_op(result, init);
	}

	template <class BlockTile, class BinaryOperation, typename T>
	GPU_DEVICE T transform_inclusive_scan(BlockTile g, T value, BinaryOperation binary_op, T init)
	{
		for (offset_t offset = 1; offset < g.size(); offset <<= 1)
		{
			T y = g.shfl_up(value, offset);
			if (g.thread_rank() >= offset)
				value = binary_op(value, y);
		}

		return binary_op(value, init);
	}
}
