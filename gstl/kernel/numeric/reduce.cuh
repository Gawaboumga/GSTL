#pragma once

#ifndef GPU_KERNEL_NUMERIC_REDUCE_HPP
#define GPU_KERNEL_NUMERIC_REDUCE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	namespace kernel
	{
		template <class RandomIt, class RandomOutputIt>
		unsigned int reduce(RandomIt first, RandomIt last, RandomOutputIt buffer);

		template <class RandomIt, class RandomOutputIt, typename T>
		unsigned int reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init);

		template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp>
		unsigned int reduce(RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op);
	}
}

#include <gstl/kernel/numeric/reduce.cu>

#endif // GPU_KERNEL_NUMERIC_REDUCE_HPP
