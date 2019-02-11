#pragma once

#ifndef GPU_ALGORITHMS_SORT_BITONIC_HPP
#define GPU_ALGORITHMS_SORT_BITONIC_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/algorithms/sort/sort_tag.cuh>

namespace gpu
{
	template <class RandomIt>
	GPU_DEVICE void bitonic_sort(block_t g, RandomIt first, RandomIt last);

	template <class RandomIt, unsigned int tile_sz>
	GPU_DEVICE void bitonic_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last);

	template <class RandomIt, class Compare>
	GPU_DEVICE void bitonic_sort(block_t g, RandomIt first, RandomIt last, Compare comp);

	template <class RandomIt, class Compare, unsigned int tile_sz>
	GPU_DEVICE void bitonic_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp);

	template <class RandomIt, class Compare, class SortTag>
	GPU_DEVICE void bitonic_sort(block_t g, RandomIt first, RandomIt last, Compare comp, SortTag tag);

	template <class RandomIt, class Compare, class SortTag, unsigned int tile_sz>
	GPU_DEVICE void bitonic_sort(block_tile_t<tile_sz> g, RandomIt first, RandomIt last, Compare comp, SortTag tag);
}

#include <gstl/algorithms/sort/bitonic_sort.cu>

#endif // GPU_ALGORITHMS_SORT_BITONIC_HPP
