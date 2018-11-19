#pragma once

#ifndef GPU_UTILITY_BALLOT_HPP
#define GPU_UTILITY_BALLOT_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	inline GPU_DEVICE bool all(block_t g, bool value);

	template <class BlockTile>
	GPU_DEVICE bool all(BlockTile g, bool value);

	inline GPU_DEVICE bool any(block_t g, bool value);

	template <class BlockTile>
	GPU_DEVICE bool any(BlockTile g, bool value);
}

#include <gstl/utility/ballot.cu>

#endif // GPU_UTILITY_BALLOT_HPP
