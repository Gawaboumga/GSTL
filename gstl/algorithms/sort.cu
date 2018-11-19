#include <gstl/algorithms/sort.cuh>

#include <gstl/functional/function_object.cuh>

namespace gpu
{
	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted(ForwardIt first, ForwardIt last)
	{
		return is_sorted(first, last, less<>());
	}

	template <class ForwardIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted(ForwardIt first, ForwardIt last, Compare comp)
	{
		return is_sorted_until(first, last, comp);
	}

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted_until(ForwardIt first, ForwardIt last)
	{
		return is_sorted_until(first, last, less<>());
	}

	template <class ForwardIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted_until(ForwardIt first, ForwardIt last, Compare comp)
	{
		if (first != last)
		{
			ForwardIt next = first;
			while (++next != last)
			{
				if (comp(*next, *first))
					return next;
				first = next;
			}
		}

		return last;
	}
}
