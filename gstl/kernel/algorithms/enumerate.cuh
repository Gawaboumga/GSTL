#pragma once

#ifndef GPU_KERNEL_ENUMERATE_HPP
#define GPU_KERNEL_ENUMERATE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class Function>
	void enumerate(RandomIt first, RandomIt last, Function f);

	template <class RandomIt, class Size, class Function>
	void enumerate(RandomIt first, Size n, Function f);
}

#include <gstl/kernel/algorithms/enumerate.cu>

#endif // GPU_KERNEL_ENUMERATE_HPP
