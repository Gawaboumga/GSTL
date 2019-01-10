#pragma once

#ifndef GPU_KERNEL_NUMERIC_TRANSFORM_REDUCE_HPP
#define GPU_KERNEL_NUMERIC_TRANSFORM_REDUCE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	namespace kernel
	{

		template <class RandomIt1, class RandomIt2, class RandomOutputIt, typename T>
		unsigned int transform_reduce(RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomOutputIt buffer, T init);

		template <class RandomIt1, class RandomIt2, class RandomOutputIt, typename T, class BinaryOp1, class BinaryOp2>
		unsigned int transform_reduce(RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomOutputIt buffer, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2);

		template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp, class UnaryOp>
		unsigned int transform_reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op, UnaryOp unary_op);
	}
}

#include <gstl/kernel/numeric/transform_reduce.cu>

#endif // GPU_KERNEL_NUMERIC_TRANSFORM_REDUCE_HPP
