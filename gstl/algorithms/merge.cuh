#pragma once

#ifndef GPU_ALGORITHMS_MERGE_HPP
#define GPU_ALGORITHMS_MERGE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class ForwardIt1, class ForwardIt2, class ForwardIt3>
	GPU_DEVICE ForwardIt3 merge(block_t g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first);

	template <class BlockTile, class ForwardIt1, class ForwardIt2, class ForwardIt3>
	GPU_DEVICE ForwardIt3 merge(BlockTile g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first);

	template <class InputIt1, class InputIt2, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt merge(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt d_first);

	template <class ForwardIt1, class ForwardIt2, class ForwardIt3, class Compare>
	GPU_DEVICE ForwardIt3 merge(block_t g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first, Compare comp);

	template <class BlockTile, class ForwardIt1, class ForwardIt2, class ForwardIt3, class Compare>
	GPU_DEVICE ForwardIt3 merge(BlockTile g, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2, ForwardIt3 d_first, Compare comp);

	template <class InputIt1, class InputIt2, class OutputIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR OutputIt merge(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt d_first, Compare comp);
}

#include <gstl/algorithms/merge.cu>

#endif // GPU_ALGORITHMS_MERGE_HPP
