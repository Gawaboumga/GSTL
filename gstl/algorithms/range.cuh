#pragma once

#ifndef GPU_ALGORITHMS_RANGE_HPP
#define GPU_ALGORITHMS_RANGE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class Thread, class Size, class Function>
	GPU_DEVICE void range(Thread g, Size stop, Function f);

	template <class Thread, class Size, class Function>
	GPU_DEVICE void range(Thread g, Size start, Size stop, Function f);

	template <class Thread, class Size, class Function>
	GPU_DEVICE void range(Thread g, Size start, Size stop, Size step, Function f);
}

#include <gstl/algorithms/range.cu>

#endif // GPU_ALGORITHMS_RANGE_HPP
