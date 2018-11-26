#include <gstl/algorithms/merge.cuh>

#include <gstl/algorithms/copy.cuh>
#include <gstl/functional/function_object.cuh>
#include <gstl/utility/pair.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Threads, class ForwardIt1, class ForwardIt2, class Compare>
		GPU_DEVICE pair<offset_t, offset_t> diagonal_intersection(Threads g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, Compare comp)
		{
			offset_t thid = g.thread_rank();
			offset_t difference_1 = distance(first1, last1);
			offset_t difference_2 = distance(first2, last2);

			offset_t chunk_size = (difference_1 + difference_2) / g.size();
			offset_t bonus = (difference_1 + difference_2) - chunk_size * g.size();

			offset_t diagonal_num = thid * chunk_size;

			if ((g.size() - bonus) <= thid)
			{
				offset_t v = thid - (g.size() - bonus);
				diagonal_num += v;
			}

			offset_t begin = 0;
			if (diagonal_num > difference_2)
				begin = diagonal_num - difference_2;
			offset_t end = min(diagonal_num, difference_1);

			if (begin >= end)
				return { begin, end };

			offset_t mid;
			while (begin < end)
			{
				mid = (begin + end) / 2u;

				if (comp(*(first1 + mid), *(first2 + diagonal_num - mid - 1)))
					begin = mid + 1;
				else
					end = mid;
			}

			if (comp(*(first1 + mid), *(first2 + diagonal_num - mid - 1)))
				return { begin, diagonal_num - mid - 1u };
			else
				return { begin, diagonal_num - mid };
		}

		template <class Threads, class ForwardIt1, class ForwardIt2, class ForwardIt3, class Compare>
		GPU_DEVICE ForwardIt3 merge(Threads g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first, Compare comp)
		{
			// Merge Path - A Visually Intuitive Approach to Parallel Merging
			offset_t thid = g.thread_rank();
			offset_t difference_1 = distance(first1, last1);
			offset_t difference_2 = distance(first2, last2);

			offset_t chunk_size = (difference_1 + difference_2) / g.size();
			offset_t bonus = (difference_1 + difference_2) - chunk_size * g.size();

			offset_t diagonal_num = thid * chunk_size;
			offset_t length = chunk_size;

			if ((g.size() - bonus) <= thid)
			{
				length += 1;
				diagonal_num += thid - (g.size() - bonus);
			}

			auto starts = diagonal_intersection(g, first1, last1, first2, last2, comp);
			merge_n(first1 + starts.first, last1,
				first2 + starts.second, last2,
				d_first + diagonal_num,
				length,
				comp);
			return d_first + difference_1 + difference_2;
		}

		template <class ForwardIt1, class ForwardIt2, class ForwardIt3, class Compare>
		GPU_DEVICE void merge_n(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first, offset_t length, Compare comp)
		{
			for (offset_t i = 0; i != length; ++i)
			{
				if (first1 == last1)
					copy_n(first2, length - i, d_first);
				else if (first2 == last2)
					copy_n(first1, length - i, d_first);
				else
				{
					if (comp(*first1, *first2))
					{
						*d_first = *first1;
						++first1;
					}
					else
					{
						*d_first = *first2;
						++first2;
					}
					++d_first;
				}
			}
		}
	}

	template <class ForwardIt1, class ForwardIt2, class ForwardIt3>
	GPU_DEVICE ForwardIt3 merge(block_t g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first)
	{
		return merge(g, first1, last1, first2, last2, d_first, less<>());
	}

	template <class BlockTile, class ForwardIt1, class ForwardIt2, class ForwardIt3>
	GPU_DEVICE ForwardIt3 merge(BlockTile g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first)
	{
		return merge(g, first1, last1, first2, last2, d_first, less<>());
	}

	template <class InputIt1, class InputIt2, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt merge(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt d_first)
	{
		return merge(first1, last1, first2, last2, d_first, less<>());
	}

	template <class ForwardIt1, class ForwardIt2, class ForwardIt3, class Compare>
	GPU_DEVICE ForwardIt3 merge(block_t g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first, Compare comp)
	{
		return detail::merge(g, first1, last1, first2, last2, d_first, comp);
	}

	template <class BlockTile, class ForwardIt1, class ForwardIt2, class ForwardIt3, class Compare>
	GPU_DEVICE ForwardIt3 merge(BlockTile g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first, Compare comp)
	{
		return detail::merge(g, first1, last1, first2, last2, d_first, comp);
	}

	template <class InputIt1, class InputIt2, class OutputIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR OutputIt merge(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt d_first, Compare comp)
	{
		for (; first1 != last1; ++d_first)
		{
			if (first2 == last2)
				return copy(first1, last1, d_first);

			if (comp(*first2, *first1))
			{
				*d_first = *first2;
				++first2;
			}
			else
			{
				*d_first = *first1;
				++first1;
			}
		}
		return copy(first2, last2, d_first);
	}
}
