#pragma once

#ifndef GPU_ALGORITHMS_DETAIL_ENUMERATE_HPP
#define GPU_ALGORITHMS_DETAIL_ENUMERATE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class Function>
		GPU_DEVICE void enumerate(Thread g, RandomIt first, RandomIt last, Function f);

		template <class Thread, class RandomIt, class Size, class Function>
		GPU_DEVICE void enumerate_n(Thread g, RandomIt first, Size n, Function f);
	}
}

#include <gstl/algorithms/detail/enumerate.cu>

#endif // GPU_ALGORITHMS_DETAIL_ENUMERATE_HPP
