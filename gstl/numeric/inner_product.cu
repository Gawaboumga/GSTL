#include <gstl/numeric/inner_product.cuh>

#include <gstl/numeric/transform_reduce.cuh>

namespace gpu
{
	template <class RandomIt1, class RandomIt2, typename T>
	GPU_DEVICE group_result<T> inner_product(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T value)
	{
		return inner_product(first1, last1, first2, value, plus<>(), multiplies<>());
	}

	template <class BlockTile, class RandomIt1, class RandomIt2, typename T>
	GPU_DEVICE group_result<T> inner_product(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T value)
	{
		return inner_product(first1, last1, first2, value, plus<>(), multiplies<>());
	}

	template <class InputIt1, class InputIt2, typename T>
	GPU_DEVICE GPU_CONSTEXPR T inner_product(InputIt1 first1, InputIt1 last1, InputIt2 first2, T value)
	{
		return inner_product(first1, last1, first2, value, plus<>(), multiplies<>());
	}

	template <class RandomIt1, class RandomIt2, typename T, class BinaryOperation1, class BinaryOperation2>
	GPU_DEVICE group_result<T> inner_product(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T value, BinaryOperation1 op1, BinaryOperation2 op2)
	{
		return transform_reduce(g, first1, last1, first2, value, op1, op2);
	}

	template <class BlockTile, class RandomIt1, class RandomIt2, typename T, class BinaryOperation1, class BinaryOperation2>
	GPU_DEVICE group_result<T> inner_product(BlockTile g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, T value, BinaryOperation1 op1, BinaryOperation2 op2)
	{
		return transform_reduce(g, first1, last1, first2, value, op1, op2);
	}

	template <class InputIt1, class InputIt2, typename T, class BinaryOperation1, class BinaryOperation2>
	GPU_DEVICE GPU_CONSTEXPR T inner_product(InputIt1 first1, InputIt1 last1, InputIt2 first2, T value, BinaryOperation1 op1, BinaryOperation2 op2)
	{
		while (first1 != last1)
		{
			value = op1(std::move(value), op2(*first1, *first2));
			++first1;
			++first2;
		}
		return value;
	}
}
