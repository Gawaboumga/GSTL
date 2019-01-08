#pragma once

#ifndef GPU_KERNEL_ALGORITHMS_FILL_HPP
#define GPU_KERNEL_ALGORITHMS_FILL_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	namespace kernel
	{
		template <class RandomIt, typename T>
		void fill(RandomIt first, RandomIt last, const T* value);
	}
}

#include <gstl/kernel/algorithms/fill.cu>

#endif // GPU_KERNEL_ALGORITHMS_FILL_HPP
