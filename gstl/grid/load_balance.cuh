#pragma once

#ifndef GPU_GRID_LOAD_BALANCE_HPP
#define GPU_GRID_LOAD_BALANCE_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/utility/pair.cuh>

namespace gpu
{
	template <class RandomIt>
	GPU_DEVICE pair<RandomIt, RandomIt> load_balance(grid_t grid, RandomIt first, RandomIt last);
}

#include <gstl/grid/load_balance.cu>

#endif // GPU_GRID_LOAD_BALANCE_HPP
