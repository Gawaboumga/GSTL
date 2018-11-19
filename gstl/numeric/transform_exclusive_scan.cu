#include <gstl/numeric/transform_exclusive_scan.cuh>

#include <gstl/functional/function_object.cuh>
#include <gstl/numeric/transform_inclusive_scan.cuh>
#include <gstl/utility/shfl.cuh>

namespace gpu
{
	namespace detail
	{
		template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
		GPU_DEVICE inline void transform_exclusive_scan_partial(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
		{
			block_tile_t<32> warp = tiled_partition<32>(g);
			if (g.thread_rank() < warp.size())
			{
				gpu::transform_exclusive_scan(warp, first, last, d_first, binary_op, unary_op, init);
			}
			g.sync();
		}

		template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
		GPU_DEVICE inline RandomOutputIt  transform_exclusive_scan_full(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
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
				*(d_first + n - 1) = init; // clear the last element

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
	GPU_DEVICE RandomOutputIt transform_exclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op)
	{
		std::result_of_t<UnaryOperation(std::remove_reference_t<decltype(*first)>)> value{};
		return transform_exclusive_scan(g, first, last, d_first, binary_op, unary_op, value);
	}

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation>
	GPU_DEVICE RandomOutputIt transform_exclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op)
	{
		std::result_of_t<UnaryOperation(std::remove_reference_t<decltype(*first)>)> value{};
		return transform_exclusive_scan(g, first, last, d_first, binary_op, unary_op, value);
	}

	template <class InputIt, class OutputIt, class BinaryOperation, class UnaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform_exclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op)
	{
		std::result_of_t<UnaryOperation(std::remove_reference_t<decltype(*first)>)> value{};
		return transform_exclusive_scan(g, first, last, d_first, binary_op, unary_op, value);
	}

	template <class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
	GPU_DEVICE RandomOutputIt transform_exclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
	{
		offset_t len = distance(first, last);
		offset_t offset = 0;

		T value = init;
		if (len < 2 * g.size())
		{
			detail::transform_exclusive_scan_partial(g, first, last, d_first, binary_op, unary_op, value);
			return d_first + len;
		}

		auto result_ptr = detail::transform_exclusive_scan_full(g, first, last, d_first, binary_op, unary_op, value);
		value = shfl(g, *result_ptr, g.size() - 1);
		value = binary_op(value, *(first + (result_ptr - d_first)));
		offset += 2 * g.size();

		while (offset < len && offset + 2 * g.size() < len)
		{
			auto result_ptr = detail::transform_exclusive_scan_full(g, first + offset, last, d_first + offset, binary_op, unary_op, value);
			value = shfl(g, *result_ptr, g.size() - 1);
			value = binary_op(value, *(first + (result_ptr - d_first)));
			offset += 2 * g.size();
		}

		if (offset < len)
			detail::transform_exclusive_scan_partial(g, first + offset, last, d_first + offset, binary_op, unary_op, value);

		return d_first + len;
	}

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, class BinaryOperation, class UnaryOperation, typename T>
	GPU_DEVICE RandomOutputIt transform_exclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
	{
		T current_value = init;
		offset_t len = distance(first, last);
		offset_t offset = 0;

		if (offset + g.thread_rank() < len)
		{
			T result = transform_exclusive_scan(g, unary_op(*(first + offset + g.thread_rank())), binary_op, current_value);
			*(d_first + offset + g.thread_rank()) = result;
			current_value = shfl(g, result, g.size() - 1);
			offset += g.size();
		}

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
	GPU_DEVICE GPU_CONSTEXPR OutputIt transform_exclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op, UnaryOperation unary_op, T init)
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
	GPU_DEVICE T transform_exclusive_scan(block_t g, T value, BinaryOperation binary_op, T init)
	{
		return 0;
	}

	template <class BlockTile, class BinaryOperation, typename T>
	GPU_DEVICE T transform_exclusive_scan(BlockTile g, T value, BinaryOperation binary_op, T init)
	{
		for (offset_t offset = 1; offset < g.size(); offset <<= 1)
		{
			T y = g.shfl_up(value, offset);
			if (g.thread_rank() > offset)
				value = binary_op(value, y);
		}

		value = binary_op(value, init);
		if (g.thread_rank() == 0)
			value = init;
		return value;
	}
}
