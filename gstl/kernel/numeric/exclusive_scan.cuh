#pragma once

#ifndef GPU_KERNEL_NUMERIC_EXCLUSIVE_SCAN_HPP
#define GPU_KERNEL_NUMERIC_EXCLUSIVE_SCAN_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class OutputIt>
	OutputIt exclusive_scan(RandomIt first, RandomIt last, OutputIt d_first);
}

#include <gstl/kernel/numeric/exclusive_scan.cu>

#endif // GPU_KERNEL_NUMERIC_EXCLUSIVE_SCAN_HPP
