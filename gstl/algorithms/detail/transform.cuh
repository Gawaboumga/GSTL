#pragma once

#ifndef GPU_ALGORITHMS_DETAIL_TRANSFORM_HPP
#define GPU_ALGORITHMS_DETAIL_TRANSFORM_HPP

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt1, class RandomIt2, class UnaryOperation>
		GPU_DEVICE void transform(Thread g, RandomIt1 first, RandomIt1 last, RandomIt2 d_first, UnaryOperation unary_op);

		template <class Thread, class RandomIt1, class RandomIt2, class RandomIt3, class BinaryOperation>
		GPU_DEVICE void transform(Thread g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt3 d_first, BinaryOperation binary_op);
	}
}

#include <gstl/algorithms/detail/transform.cu>

#endif // GPU_ALGORITHMS_DETAIL_TRANSFORM_HPP
