#pragma once

#ifndef GPU_KERNEL_NUMERIC_REDUCE_HPP
#define GPU_KERNEL_NUMERIC_REDUCE_HPP

#include <gstl/prerequisites.hpp>

#include <cuda-api-wrappers/cuda/api_wrappers.hpp>

namespace gpu
{
	namespace kernel
	{
		template <class RandomIt>
		typename std::iterator_traits<RandomIt>::value_type reduce(RandomIt first, RandomIt last);

		template <class RandomIt, typename T>
		T reduce(RandomIt first, RandomIt last, T init);

		template <class RandomIt, typename T, class BinaryOp>
		T reduce(RandomIt first, RandomIt last, T init, BinaryOp binary_op);

		template <class RandomIt>
		typename std::iterator_traits<RandomIt>::value_type reduce(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last);

		template <class RandomIt, typename T>
		T reduce(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, T init);

		template <class RandomIt, typename T, class BinaryOp>
		T reduce(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, T init, BinaryOp binary_op);

		template <class RandomIt, class RandomOutputIt>
		void reduce_to_buffer(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, RandomOutputIt buffer);

		template <class RandomIt, class RandomOutputIt, typename T>
		void reduce_to_buffer(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, RandomOutputIt buffer, T init);

		template <class RandomIt, class RandomOutputIt, typename T, class BinaryOp>
		void reduce_to_buffer(cuda::launch_configuration_t configuration, RandomIt first, RandomIt last, RandomOutputIt buffer, T init, BinaryOp binary_op);
	}
}

#include <gstl/kernel/numeric/reduce.cu>

#endif // GPU_KERNEL_NUMERIC_REDUCE_HPP
