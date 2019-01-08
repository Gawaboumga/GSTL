#include <gstl/grid/numeric/reduce.cuh>

#include <gstl/functional/function_object.cuh>
#include <gstl/grid/numeric/transform_reduce.cuh>

#include <iterator>

namespace gpu
{
	template <class RandomIt, class RandomOutputIt>
	GPU_DEVICE void reduce(grid_t g, RandomIt first, RandomIt last, RandomOutputIt buffer)
	{
		return reduce(g, first, last, buffer, typename std::iterator_traits<RandomIt>::value_type{});
	}

	template <class RandomIt, class RandomOutputIt, typename T>
	GPU_DEVICE void reduce(grid_t g, RandomIt first, RandomIt last, RandomOutputIt buffer, T init)
	{
		return reduce(g, first, last, buffer, init, plus<>());
	}

	template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp>
	GPU_DEVICE void reduce(grid_t g, RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op)
	{
		return transform_reduce(g, first, last, buffer, init, binary_op, identity());
	}
}
