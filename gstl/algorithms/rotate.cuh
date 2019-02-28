#pragma once

#ifndef GPU_ALGORITHMS_ROTATE_HPP
#define GPU_ALGORITHMS_ROTATE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class ForwardIt>
	GPU_DEVICE ForwardIt rotate(block_t g, ForwardIt first, ForwardIt n_first, ForwardIt last);

	template <class BlockTile, class ForwardIt>
	GPU_DEVICE ForwardIt rotate(BlockTile g, ForwardIt first, ForwardIt n_first, ForwardIt last);

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt rotate(ForwardIt first, ForwardIt n_first, ForwardIt last);
}

#include <gstl/algorithms/rotate.cu>

#endif // GPU_ALGORITHMS_ROTATE_HPP
