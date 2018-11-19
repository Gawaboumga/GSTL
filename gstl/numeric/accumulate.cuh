#pragma once

#ifndef GPU_NUMERIC_ACCUMULATE_HPP
#define GPU_NUMERIC_ACCUMULATE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class BlockTile, typename T>
	GPU_DEVICE T accumulate(BlockTile g, T value, T init);

	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR T accumulate(InputIt first, InputIt last, T init);

	template <class BlockTile, typename T>
	GPU_DEVICE T accumulate(BlockTile g, T value, T init, BinaryOperation op);

	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR T accumulate(InputIt first, InputIt last, T init, BinaryOperation op);
}

#include <gstl/numeric/accumulate.cu>

#endif // GPU_NUMERIC_ACCUMULATE_HPP
