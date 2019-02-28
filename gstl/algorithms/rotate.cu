#include <gstl/algorithms/rotate.cuh>

#include <gstl/algorithms/move.cuh>
#include <gstl/algorithms/swap.cuh>
#include <gstl/utility/iterator.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt>
		GPU_DEVICE RandomIt rotate_left(Thread g, RandomIt first, RandomIt last)
		{
			typename std::iterator_traits<RandomIt>::value_type tmp;
			if (g.thread_rank() == 0)
				tmp = std::move(*first);
			g.sync();

			auto lm1 = move(g, next(first), last, first);

			if (g.thread_rank() == 0)
				*lm1 = std::move(tmp);

			return lm1;
		}

		template <class Thread, class RandomIt>
		GPU_DEVICE RandomIt rotate_right(Thread g, RandomIt first, RandomIt last)
		{
			auto lm1 = prev(last);

			typename std::iterator_traits<RandomIt>::value_type tmp;
			if (g.thread_rank() == 0)
				tmp = std::move(*lm1);
			g.sync();

			auto fp1 = move_backward(g, first, lm1, last);

			if (g.thread_rank() == 0)
				*first = std::move(tmp);

			return fp1;
		}

		template <class Thread, class RandomIt>
		GPU_DEVICE RandomIt rotate(Thread g, RandomIt first, RandomIt n_first, RandomIt last)
		{
			if (next(first) == n_first)
				return rotate_left(g, first, last);

			if (next(n_first) == last)
				return rotate_right(g, first, last);

			// gries and mills block swap
			auto n = distance(first, last);
			auto d = distance(first, n_first);

			auto i = d;
			auto j = n - d;

			while (i != j)
			{
				if (i < j)
				{
					swap_ranges_n(g, first + d - i, first + d + j - i, i);
					j -= i;
				}
				else
				{
					swap_ranges_n(g, first + d - i, first + d, j);
					i -= j;
				}
				g.sync();
			}

			swap_ranges_n(g, first + d - i, first + d, i);

			return first + (last - n_first);
		}
	}

	template <class ForwardIt>
	GPU_DEVICE ForwardIt rotate(block_t g, ForwardIt first, ForwardIt n_first, ForwardIt last)
	{
		if (first == n_first)
			return last;

		if (n_first == last)
			return first;

		return detail::rotate(g, first, n_first, last);
	}

	template <class BlockTile, class ForwardIt>
	GPU_DEVICE ForwardIt rotate(BlockTile g, ForwardIt first, ForwardIt n_first, ForwardIt last)
	{
		if (first == n_first)
			return last;

		if (n_first == last)
			return first;

		return detail::rotate(g, first, n_first, last);
	}

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt rotate(ForwardIt first, ForwardIt n_first, ForwardIt last)
	{
		if (first == n_first)
			return last;

		if (n_first == last)
			return first;

		ForwardIt first2 = n_first;

		do
		{
			iter_swap(first, first2);
			++first;
			++first2;
			if (first == new_first)
				new_first = first2;
		} while (first2 != last);

		ForwardIt ret = first;
		first2 = n_first;

		while (first2 != last)
		{
			iter_swap(first, first2);
			++first;
			++first2;
			if (first == new_first)
				new_first = first2;
			else if (first2 == last)
				first2 = new_first;
		}

		return ret;
	}
}
