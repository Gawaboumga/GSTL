#pragma once

#ifndef GPU_UTILITY_SWAP_HPP
#define GPU_UTILITY_SWAP_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class ForwardIt1, class ForwardIt2>
	GPU_DEVICE GPU_CONSTEXPR void iter_swap(ForwardIt1 a, ForwardIt2 b);

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR void swap(T& a, T& b) noexcept;

	template <typename T, std::size_t N>
	GPU_DEVICE GPU_CONSTEXPR void swap(T (&a)[N], T (&b)[N]) noexcept;
}

#include <gstl/utility/swap.cu>

#endif // GPU_UTILITY_SWAP_HPP
