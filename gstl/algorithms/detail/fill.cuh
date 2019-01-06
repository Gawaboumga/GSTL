#pragma once

#ifndef GPU_ALGORITHMS_DETAIL_FILL_HPP
#define GPU_ALGORITHMS_DETAIL_FILL_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, typename T>
		GPU_DEVICE void fill(Thread g, RandomIt first, RandomIt last, const T& value);

		template <class Thread, class ForwardIt, class Size, typename T>
		GPU_DEVICE void fill_n(Thread g, ForwardIt first, Size n, const T& value);
	}
}

#include <gstl/algorithms/detail/fill.cu>

#endif // GPU_ALGORITHMS_DETAIL_FILL_HPP
