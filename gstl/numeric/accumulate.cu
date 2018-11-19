#include <gstl/numeric/accumulate.cuh>

#include <gstl/numeric/reduce.cuh>

namespace gpu
{
	template <class BlockTile, typename T>
	GPU_DEVICE T accumulate(block_tile_t<tile_size> g, T value, T init)
	{
		return accumulate(g, value, init, plus<>());
	}

	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR T accumulate(InputIt first, InputIt last, T init)
	{
		return accumulate(first, last, init, plus<>());
	}

	template <class BlockTile, typename T>
	GPU_DEVICE T accumulate(BlockTile g, T value, T init, BinaryOperation op)
	{
		if (g.thread_rank() == 0)
			value = op(std::move(init), value);

		for (offset_t i = g.size() / 2; i > 0; i /= 2)
		{
			value = op(value, g.shfl_down(value, i));
		}
		return value;
	}

	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR T accumulate(InputIt first, InputIt last, T init, BinaryOperation op)
	{
		for (; first != last; ++first)
		{
			init = op(std::move(init), *first);
		}
		return init;
	}
}
