#include <gstl/algorithms/binary_search.cuh>

#include <gstl/functional/function_object.cuh>
#include <gstl/math/intrinsics.cuh>
#include <gstl/utility/ballot.cuh>
#include <gstl/utility/iterator.cuh>

namespace gpu
{
	namespace detail
	{
		template <typename T>
		GPU_DEVICE const T& min(const T& lhs, const T& rhs)
		{
			return lhs < rhs ? lhs : rhs;
		}

		template <class Threads, class RandomIt, typename T, class BinaryPredicate>
		GPU_DEVICE RandomIt lower_bound(Threads g, RandomIt first, RandomIt last, const T& value, BinaryPredicate p)
		{
			RandomIt it;
			typename std::iterator_traits<RandomIt>::difference_type count, step;
			count = distance(first, last);

			while (count > g.size())
			{
				it = first;
				step = count / 2;
				unsigned int shift = gpu::ffs(g.size()) - 1;
				step >>= shift;
				step <<= shift;
				it += step;

				bool result = false;
				if (it + g.thread_rank() < last)
					result = p(*(it + g.thread_rank()), value);

				if (all(g, result))
				{
					first = it + g.size();
					count -= step + g.size();
				}
				else if (all(g, !result))
					count = step;
				else
					return it + first_index(g, !result);
			}

			it = first;
			bool result = false;
			if (it + g.thread_rank() < last)
				result = p(*(it + g.thread_rank()), value);

			if (all(g, result))
				return min(it + g.size(), last);
			else if (all(g, !result))
				return it;
			else
				return min(it + first_index(g, !result), last);
		}
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE bool binary_search(block_t g, ForwardIt first, ForwardIt last, const T& value)
	{
		return binary_search(g, first, last, value, less<>());
	}

	template <class BlockTile, class ForwardIt, typename T>
	GPU_DEVICE bool binary_search(BlockTile g, ForwardIt first, ForwardIt last, const T& value)
	{
		return binary_search(g, first, last, value, less<>());
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR bool binary_search(ForwardIt first, ForwardIt last, const T& value)
	{
		return binary_search(first, last, value, less<>());
	}

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE bool binary_search(block_t g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		first = lower_bound(g, first, last, value, p);
		return (!(first == last) && !(p(value < *first)));
	}

	template <class BlockTile, class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE bool binary_search(BlockTile g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		first = lower_bound(g, first, last, value, p);
		return (!(first == last) && !(p(value < *first)));
	}

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool binary_search(ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		first = lower_bound(first, last, value, p);
		return (!(first == last) && !(p(value < *first)));
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE pair<ForwardIt, ForwardIt> equal_range(block_t g, ForwardIt first, ForwardIt last, const T& value)
	{
		return equal_range(g, first, last, value, less<>());
	}

	template <class BlockTile, class ForwardIt, typename T>
	GPU_DEVICE pair<ForwardIt, ForwardIt> equal_range(BlockTile g, ForwardIt first, ForwardIt last, const T& value)
	{
		return equal_range(g, first, last, value, less<>());
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR pair<ForwardIt, ForwardIt> equal_range(ForwardIt first, ForwardIt last, const T& value)
	{
		return equal_range(first, last, value, less<>());
	}

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE pair<ForwardIt, ForwardIt> equal_range(block_t g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class BlockTile, class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE pair<ForwardIt, ForwardIt> equal_range(BlockTile g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p);

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR pair<ForwardIt, ForwardIt> equal_range(ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		return gpu::make_pair(
			lower_bound(first, last, value, p),
			upper_bound(first, last, value, p)
		);
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE ForwardIt lower_bound(block_t g, ForwardIt first, ForwardIt last, const T& value)
	{
		return lower_bound(g, first, last, value, less<>());
	}

	template <class BlockTile, class ForwardIt, typename T>
	GPU_DEVICE ForwardIt lower_bound(BlockTile g, ForwardIt first, ForwardIt last, const T& value)
	{
		return lower_bound(g, first, last, value, less<>());
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value)
	{
		return lower_bound(first, last, value, less<>());
	}

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE ForwardIt lower_bound(block_t g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		return detail::lower_bound(g, first, last, value, p);
	}

	template <class BlockTile, class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE ForwardIt lower_bound(BlockTile g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		return detail::lower_bound(g, first, last, value, p);
	}

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		ForwardIt it;
		typename std::iterator_traits<ForwardIt>::difference_type count, step;
		count = distance(first, last);

		while (count > 0)
		{
			it = first;
			step = count / 2;
			advance(it, step);
			if (comp(*it, value))
			{
				first = ++it;
				count -= step + 1;
			}
			else
				count = step;
		}
		return first;
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE ForwardIt upper_bound(block_t g, ForwardIt first, ForwardIt last, const T& value)
	{
		return upper_bound(g, first, last, value, less<>());
	}

	template <class BlockTile, class ForwardIt, typename T>
	GPU_DEVICE ForwardIt upper_bound(BlockTile g, ForwardIt first, ForwardIt last, const T& value)
	{
		return upper_bound(g, first, last, value, less<>());
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value)
	{
		return upper_bound(g, first, last, value, less<>());
	}

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE ForwardIt upper_bound(block_t g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		return lower_bound(g, first, last, value, [&p](const auto& lhs, const auto& rhs) {
			return !p(lhs, rhs);
		});
	}

	template <class BlockTile, class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE ForwardIt upper_bound(BlockTile g, ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		return lower_bound(g, first, last, value, [&p](const auto& lhs, const auto& rhs) {
			return !p(lhs, rhs);
		});
	}

	template <class ForwardIt, typename T, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, BinaryPredicate p)
	{
		return lower_bound(first, last, value, [&p](const auto& lhs, const auto& rhs) {
			return !p(lhs, rhs);
		});
	}
}
