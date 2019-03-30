#include <gstl/algorithms/inplace_merge.cuh>

#include <gstl/algorithms/count.cuh>
#include <gstl/algorithms/move.cuh>
#include <gstl/functional/function_object.cuh>
#include <gstl/utility/pair.cuh>

namespace gpu
{
	namespace detail
	{
		template <class Threads, class BidirIt, class Compare>
		GPU_DEVICE BidirIt diagonal_intersection(Threads g, offset_t thread_offset, BidirIt first, BidirIt middle, BidirIt last, Compare comp)
		{
			offset_t thid = g.thread_rank();
			offset_t difference_1 = distance(first, middle);
			offset_t difference_2 = distance(middle, last);

			offset_t diagonal_num = thread_offset + thid;

			offset_t begin = 0;
			if (thid >= distance(first, last))
				return last;
			else if (diagonal_num > difference_2)
				begin = diagonal_num - difference_2;
			offset_t end = min(diagonal_num, difference_1);

			offset_t middle_point = 0;
			while (begin < end)
			{
				middle_point = (begin + end) / 2u;

				if (!comp(*(middle + diagonal_num - middle_point - 1), *(first + middle_point)))
					begin = middle_point + 1;
				else
					end = middle_point;
			}

			auto cmp1 = first + middle_point;
			BidirIt cmp2;
			if (diagonal_num == 0)
				cmp2 = middle + diagonal_num - middle_point; // Is equal to middle
			else
				cmp2 = middle + diagonal_num - middle_point - 1;

			if (!comp(*cmp2, *cmp1))
			{
				if (begin == difference_1)
					return cmp2;
				else if (diagonal_num - middle_point - 1 == difference_2)
					return first + begin;
				else if (!comp(*cmp2, *(first + begin)))
					return first + begin;
				else
					return cmp2;
			}
			else
			{
				auto it = middle + diagonal_num - middle_point;
				if (begin == difference_1)
					return it;
				else if (diagonal_num - middle_point == difference_2)
					return first + begin;
				else if (!comp(*it, *(first + begin)))
					return first + begin;
				else
					return it;
			}
		}

		template <class Threads, class BidirIt, class Compare>
		GPU_DEVICE void inplace_merge(Threads g, BidirIt first, BidirIt middle, BidirIt last, Compare comp)
		{
			while (first != middle && middle != last)
			{
				offset_t thid = g.thread_rank();
				offset_t difference_1 = distance(first, middle);
				offset_t total_size = distance(first, last);
				offset_t thread_offset;
				if (total_size > g.size())
					thread_offset = total_size - g.size();
				else
					thread_offset = 0;

				offset_t number_of_active_threads = total_size - thread_offset;
				auto position = diagonal_intersection(g, thread_offset, first, middle, last, comp);

				offset_t number_of_as = count(g, position < middle);
				offset_t number_of_bs = number_of_active_threads - number_of_as;

				if (number_of_as == 0)
				{
					last = last - number_of_active_threads;
					continue;
				}

				typename std::decay<decltype(*first)>::type value;
				if (thid < number_of_active_threads)
					value = std::move(*position);
				g.sync();

				move(g, middle, last - number_of_bs, middle - number_of_as);
				g.sync();

				if (thid < number_of_active_threads)
					*(first + thread_offset + thid) = std::move(value);
				g.sync();

				middle = middle - number_of_as;
				last = last - number_of_active_threads;
			}
		}
	}

	template <class BidirIt>
	GPU_DEVICE void inplace_merge(block_t g, BidirIt first, BidirIt middle, BidirIt last)
	{
		return inplace_merge(g, first, middle, last, less<>());
	}

	template <class BlockTile, class BidirIt>
	GPU_DEVICE void inplace_merge(BlockTile g, BidirIt first, BidirIt middle, BidirIt last)
	{
		return inplace_merge(g, first, middle, last, less<>());
	}

	template <class BidirIt>
	GPU_DEVICE GPU_CONSTEXPR void inplace_merge(BidirIt first, BidirIt middle, BidirIt last)
	{
		return inplace_merge(first, middle, last, less<>());
	}

	template <class BidirIt, class Compare>
	GPU_DEVICE void inplace_merge(block_t g, BidirIt first, BidirIt middle, BidirIt last, Compare comp)
	{
		detail::inplace_merge(g, first, middle, last, comp);
	}

	template <class BlockTile, class BidirIt, class Compare>
	GPU_DEVICE void inplace_merge(BlockTile g, BidirIt first, BidirIt middle, BidirIt last, Compare comp)
	{
		detail::inplace_merge(g, first, middle, last, comp);
	}

	template <class BidirIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR void inplace_merge(BidirIt first, BidirIt middle, BidirIt last, Compare comp)
	{
		if (first == middle || middle == last)
			return;

		--middle;
		--last;

		while (first != middle && middle != last)
		{
			if (!comp(*last, *middle))
			{
				iter_swap(middle, last);
				--last;
			}
			else
			{
				iter_swap(middle, last);
				--middle;
			}
		}
	}
}
