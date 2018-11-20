#include <gstl/algorithms/find.cuh>

#include <gstl/utility/ballot.cuh>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE RandomIt find(block_t g, RandomIt first, RandomIt last, const T& value)
	{
		return find_if(g, first, last, [&value](decltype(*first) i) {
			return i == value;
		});
	}

	template <class BlockTile, class RandomIt, typename T>
	GPU_DEVICE RandomIt find(BlockTile g, RandomIt first, RandomIt last, const T& value)
	{
		return find_if(g, first, last, [&value](decltype(*first) i) {
			return i == value;
		});
	}

	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR InputIt find(InputIt first, InputIt last, const T& value)
	{
		return find_if(first, last, [&value](decltype(*first) i) {
			return i == value;
		});
	}

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE RandomIt find_if(block_t g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		do
		{
			bool result = false;
			if (offset + thid < len)
				result = p(*(first + offset + thid));

			offset_t index = first_index(g, result);
			if (index != g.size())
				return first + offset + index;

			offset += g.size();
		} while (offset + g.size() < len);

		return last;
	}

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE RandomIt find_if(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		offset_t len = distance(first, last);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;

		do
		{
			bool result = false;
			if (offset + thid < len)
				result = p(*(first + offset + thid));

			offset_t index = first_index(g, result);
			if (index != g.size())
				return first + offset + index;

			offset += g.size();
		} while (offset + g.size() < len);

		return last;
	}

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR InputIt find_if(InputIt first, InputIt last, UnaryPredicate p)
	{
		for (; first != last; ++first)
			if (p(*first))
				return first;

		return last;
	}

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE RandomIt find_if_not(block_t g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		return find_if(g, first, last, [&p](decltype(*first) value) {
			return !p(value);
		});
	}

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE RandomIt find_if_not(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		return find_if(g, first, last, [&p](decltype(*first) value) {
			return !p(value);
		});
	}

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR InputIt find_if_not(InputIt first, InputIt last, UnaryPredicate p)
	{
		return find_if(first, last, [&p](decltype(*first) value) {
			return !p(value);
		});
	}

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool all_of(block_t g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		return find_if_not(g, first, last, p) == last;
	}

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool all_of(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		return find_if_not(g, first, last, p) == last;
	}

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool all_of(InputIt first, InputIt last, UnaryPredicate p)
	{
		return find_if_not(first, last, p) == last;
	}

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool any_of(block_t g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		return find_if(g, first, last, p) != last;
	}

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool any_of(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		return find_if(g, first, last, p) != last;
	}

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool any_of(InputIt first, InputIt last, UnaryPredicate p)
	{
		return find_if(first, last, p) != last;
	}

	template <class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool none_of(block_t g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		return find_if(g, first, last, p) == last;
	}

	template <class BlockTile, class RandomIt, class UnaryPredicate>
	GPU_DEVICE bool none_of(BlockTile g, RandomIt first, RandomIt last, UnaryPredicate p)
	{
		return find_if(g, first, last, p) == last;
	}

	template <class InputIt, class UnaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool none_of(InputIt first, InputIt last, UnaryPredicate p)
	{
		return find_if(first, last, p) == last;
	}
}
