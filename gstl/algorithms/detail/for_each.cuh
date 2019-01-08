#pragma once

#ifndef GPU_ALGORITHMS_DETAIL_FOR_EACH_HPP
#define GPU_ALGORITHMS_DETAIL_FOR_EACH_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class UnaryFunction>
		GPU_DEVICE void for_each(Thread g, RandomIt first, RandomIt last, UnaryFunction unary_op);

		template <class Thread, class ForwardIt, class Size, class UnaryFunction>
		GPU_DEVICE void for_each_n(Thread g, ForwardIt first, Size n, UnaryFunction unary_op);
	}
}

#include <gstl/algorithms/detail/for_each.cu>

#endif // GPU_ALGORITHMS_DETAIL_FOR_EACH_HPP
