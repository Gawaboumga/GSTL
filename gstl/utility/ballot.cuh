#pragma once

#ifndef GPU_UTILITY_BALLOT_HPP
#define GPU_UTILITY_BALLOT_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	GPU_DEVICE inline bool all(block_t g, bool value);

	template <class BlockTile>
	GPU_DEVICE inline bool all(BlockTile g, bool value);

	GPU_DEVICE inline bool any(block_t g, bool value);

	template <class BlockTile>
	GPU_DEVICE inline bool any(BlockTile g, bool value);

	GPU_DEVICE inline unsigned int count(block_t g, bool value);

	template <class BlockTile>
	GPU_DEVICE inline unsigned int count(BlockTile g, bool value);

	GPU_DEVICE inline offset_t first_index(block_t g, bool value, offset_t from = 0u);

	template <class BlockTile>
	GPU_DEVICE inline offset_t first_index(BlockTile g, bool value);

	template <class BlockTile>
	GPU_DEVICE inline offset_t first_index(BlockTile g, bool value, offset_t from);

	template <typename T>
	GPU_DEVICE inline T shfl(block_t g, T value, unsigned int thid = 0u);

	template <class BlockTile, typename T>
	GPU_DEVICE inline T shfl(BlockTile g, T value, unsigned int thid = 0u);
}

#include <gstl/utility/ballot.cu>

#endif // GPU_UTILITY_BALLOT_HPP
