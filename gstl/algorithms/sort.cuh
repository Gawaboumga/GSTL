#pragma once

#ifndef GPU_ALGORITHMS_SORT_HPP
#define GPU_ALGORITHMS_SORT_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted(ForwardIt first, ForwardIt last);

	template <class ForwardIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted(ForwardIt first, ForwardIt last, Compare comp);

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted_until(ForwardIt first, ForwardIt last);

	template <class ForwardIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted_until(ForwardIt first, ForwardIt last, Compare comp);
}

#include <gstl/algorithms/sort.cu>

#endif // GPU_ALGORITHMS_SORT_HPP
