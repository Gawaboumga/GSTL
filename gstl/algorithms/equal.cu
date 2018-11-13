#include <gstl/algorithms/equal.cuh>

#include <gstl/functional/function_object.cuh>
#include <gstl/utility/ballot.cuh>

namespace gpu
{
	template <class RandomIt1, class RandomIt2>
	GPU_DEVICE bool equal(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2)
	{
		return equal(g, first1, last1, first2, equal_to<>());
	}

	template <class RandomIt1, class RandomIt2, int tile_size>
	GPU_DEVICE bool equal(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2)
	{
		return equal(g, first1, last1, first2, equal_to<>());
	}

	template <class InputIt1, class InputIt2>
	GPU_DEVICE GPU_CONSTEXPR bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2)
	{
		return equal(first1, last1, first2, equal_to<>());
	}

	template <class RandomIt1, class RandomIt2, class BinaryPredicate>
	GPU_DEVICE bool equal(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, BinaryPredicate p)
	{
		offset_t len = distance(first1, last1);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;
		bool result = true;

		while (result && offset < len)
		{
			bool thread_result;
			if (offset + thid < len)
				thread_result = p(*(first1 + offset + thid), *(first2 + offset + thid));
			else
				thread_result = true;
			result = all(g, thread_result);
			offset += g.size();
		}

		return result;
	}

	template <class RandomIt1, class RandomIt2, class BinaryPredicate, int tile_size>
	GPU_DEVICE bool equal(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, BinaryPredicate p)
	{
		offset_t len = distance(first1, last1);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;
		bool result = true;

		while (result && offset < len)
		{
			bool thread_result;
			if (offset + thid < len)
				thread_result = p(*(first1 + offset + thid), *(first2 + offset + thid));
			else
				thread_result = true;
			result = all(g, thread_result);
			offset += g.size();
		}

		return result;
	}

	template <class InputIt1, class InputIt2, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryPredicate p)
	{
		for (; first1 != last1; ++first1, ++first2)
			if (!p(*first1, *first2))
				return false;

		return true;
	}

	template <class RandomIt1, class RandomIt2>
	GPU_DEVICE bool equal(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2)
	{
		return equal(g, first1, last1, first2, last2, equal_to<>());
	}

	template <class RandomIt1, class RandomIt2, int tile_size>
	GPU_DEVICE bool equal(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2)
	{
		return equal(g, first1, last1, first2, last2, equal_to<>());
	}

	template <class InputIt1, class InputIt2>
	GPU_DEVICE GPU_CONSTEXPR bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2)
	{
		return equal(g, first1, last1, first2, last2, equal_to<>());
	}

	template <class RandomIt1, class RandomIt2, class BinaryPredicate>
	GPU_DEVICE bool equal(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2, BinaryPredicate p)
	{
		offset_t len1 = distance(first1, last1);
		offset_t len2 = distance(first2, last2);
		offset_t len = min(len1, len2);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;
		bool result = true;

		while (result && offset < len)
		{
			bool thread_result;
			if (offset + thid < len)
				thread_result = p(*(first1 + offset + thid), *(first2 + offset + thid));
			else
				thread_result = true;
			result = all(g, thread_result);
			offset += g.size();
		}

		return result;
	}

	template <class RandomIt1, class RandomIt2, class BinaryPredicate, int tile_size>
	GPU_DEVICE bool equal(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2, BinaryPredicate p)
	{
		offset_t len1 = distance(first1, last1);
		offset_t len2 = distance(first2, last2);
		offset_t len = min(len1, len2);
		offset_t thid = g.thread_rank();
		offset_t offset = 0;
		bool result = true;

		while (result && offset < len)
		{
			bool thread_result;
			if (offset + thid < len)
				thread_result = p(*(first1 + offset + thid), *(first2 + offset + thid));
			else
				thread_result = true;
			result = all(g, thread_result);
			offset += g.size();
		}

		return result;
	}

	template <class InputIt1, class InputIt2, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, BinaryPredicate p)
	{
		for (; first1 != last1 && first2 != last2; ++first1, ++first2)
			if (!p(*first1, *first2))
				return false;

		return first1 == last1 && first2 == last2;
	}
}
