#pragma once

#ifndef GPU_ALGORITHMS_EQUAL_HPP
#define GPU_ALGORITHMS_EQUAL_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class RandomIt1, class RandomIt2>
	GPU_DEVICE bool equal(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2);

	template <class RandomIt1, class RandomIt2, int tile_size>
	GPU_DEVICE bool equal(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2);

	template <class InputIt1, class InputIt2>
	GPU_DEVICE GPU_CONSTEXPR bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2);

	template <class RandomIt1, class RandomIt2, class BinaryPredicate>
	GPU_DEVICE bool equal(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, BinaryPredicate p);

	template <class RandomIt1, class RandomIt2, class BinaryPredicate, int tile_size>
	GPU_DEVICE bool equal(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, BinaryPredicate p);

	template <class InputIt1, class InputIt2, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryPredicate p);

	template <class RandomIt1, class RandomIt2>
	GPU_DEVICE bool equal(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2);

	template <class RandomIt1, class RandomIt2, int tile_size>
	GPU_DEVICE bool equal(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2);

	template <class InputIt1, class InputIt2>
	GPU_DEVICE GPU_CONSTEXPR bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

	template <class RandomIt1, class RandomIt2, class BinaryPredicate>
	GPU_DEVICE bool equal(block_t g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2, BinaryPredicate p);

	template <class RandomIt1, class RandomIt2, class BinaryPredicate, int tile_size>
	GPU_DEVICE bool equal(block_tile_t<tile_size> g, RandomIt1 first1, RandomIt1 last1, RandomIt2 first2, RandomIt2 last2, BinaryPredicate p);

	template <class InputIt1, class InputIt2, class BinaryPredicate>
	GPU_DEVICE GPU_CONSTEXPR bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, BinaryPredicate p);
}

#include <gstl/algorithms/equal.cu>

#endif // GPU_ALGORITHMS_EQUAL_HPP
