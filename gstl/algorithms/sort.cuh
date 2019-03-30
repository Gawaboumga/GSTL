#pragma once

#ifndef GPU_ALGORITHMS_SORT_HPP
#define GPU_ALGORITHMS_SORT_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt>
	GPU_DEVICE void sort(block_t g, RandomIt first, RandomIt last);

	template <class BlockTile, class RandomIt>
	GPU_DEVICE void sort(BlockTile g, RandomIt first, RandomIt last);

	template <class RandomIt, class Compare>
	GPU_DEVICE void sort(block_t g, RandomIt first, RandomIt last, Compare comp);

	template <class BlockTile, class RandomIt, class Compare>
	GPU_DEVICE void sort(BlockTile g, RandomIt first, RandomIt last, Compare comp);
}

#include <gstl/algorithms/sort.cu>

#endif // GPU_ALGORITHMS_SORT_HPP
