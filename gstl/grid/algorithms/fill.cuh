#pragma once

#ifndef GPU_GRID_FILL_HPP
#define GPU_GRID_FILL_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, typename T>
	GPU_DEVICE void fill(grid_t g, RandomIt first, RandomIt last, const T& value);
}

#include <gstl/grid/algorithms/fill.cu>

#endif // GPU_GRID_FILL_HPP
