#include <gstl/numeric/exclusive_scan.cuh>

#include <gstl/functional/function_object.cuh>
#include <gstl/numeric/transform_exclusive_scan.cuh>

namespace gpu
{
	template <class RandomInputIt, class RandomOutputIt, typename T>
	GPU_DEVICE RandomOutputIt exclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, T init)
	{
		return exclusive_scan(g, first, last, d_first, init, plus<>());
	}

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, typename T>
	GPU_DEVICE RandomOutputIt exclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, T init)
	{
		return exclusive_scan(g, first, last, d_first, init, plus<>());
	}

	template <class InputIt, class OutputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init)
	{
		return exclusive_scan(first, last, d_first, init, plus<>());
	}

	template <class RandomInputIt, class RandomOutputIt, typename T, class BinaryOperation>
	GPU_DEVICE RandomOutputIt exclusive_scan(block_t g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, T init, BinaryOperation binary_op)
	{
		return transform_exclusive_scan(g, first, last, d_first, binary_op, identity(), init);
	}

	template <class BlockTile, class RandomInputIt, class RandomOutputIt, typename T, class BinaryOperation>
	GPU_DEVICE RandomOutputIt exclusive_scan(BlockTile g, RandomInputIt first, RandomInputIt last, RandomOutputIt d_first, T init, BinaryOperation binary_op)
	{
		return transform_exclusive_scan(g, first, last, d_first, binary_op, identity(), init);
	}

	template <class InputIt, class OutputIt, typename T, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init, BinaryOperation binary_op)
	{
		return transform_exclusive_scan(g, first, last, d_first, binary_op, identity(), init);
	}

	template <class BlockTile, typename T>
	GPU_DEVICE T exclusive_scan(BlockTile g, T value, T init)
	{
		return exclusive_scan(g, value, init, plus<>());
	}

	template <class BlockTile, typename T, class BinaryOperation>
	GPU_DEVICE T exclusive_scan(BlockTile g, T value, T init, BinaryOperation binary_op)
	{
		return transform_exclusive_scan(g, value, binary_op, init);
	}
}
