#include <gstl/algorithms/count.cuh>

#include <gstl/numeric/transform_reduce.cuh>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE count_return_type<RandomIt> count(block_t g, RandomIt first, RandomIt last, const T& value)
	{
		return count_if(g, first, last, [&value](const T& data) -> bool {
			return data == value;
		});
	}

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE count_return_type<RandomIt> count(BlockTile g, RandomIt first, RandomIt last, const T& value)
	{
		return count_if(g, first, last, [&value](const T& data) -> bool {
			return data == value;
		});
	}

	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR count_return_type<InputIt> count(InputIt first, InputIt last, const T& value)
	{
		return count_if(g, first, last, [&value](const T& data) -> bool {
			return data == value;
		});
	}

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE count_return_type<RandomIt> count_if(block_t g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		typename std::iterator_traits<RandomIt>::difference_type default_value{};
		return transform_reduce(g, first, last, default_value, plus<>(), p);
	}

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE count_return_type<RandomIt> count_if(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		typename std::iterator_traits<RandomIt>::difference_type default_value{};
		return transform_reduce(g, first, last, default_value, plus<>(), p);
	}

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR count_return_type<InputIt> count_if(InputIt first, InputIt last, UnaryPredicate p)
	{
		typename std::iterator_traits<InputIt>::difference_type default_value{};
		for (; first != last; ++first) {
			if (p(*first)) {
				++ret;
			}
		}
		return ret;
	}
}
