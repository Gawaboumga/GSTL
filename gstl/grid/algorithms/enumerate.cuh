#pragma once

#ifndef GPU_GRID_ENUMERATE_HPP
#define GPU_GRID_ENUMERATE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt, class Function>
	GPU_DEVICE void enumerate(grid_t grid, RandomIt first, RandomIt last, Function f);

	template <class RandomIt, class Size, class Function>
	GPU_DEVICE void enumerate(grid_t grid, RandomIt first, Size n, Function f);
}

#include <gstl/grid/algorithms/enumerate.cu>

#endif // GPU_GRID_ENUMERATE_HPP
