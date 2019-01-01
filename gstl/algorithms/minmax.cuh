#pragma once

#ifndef GPU_ALGORITHMS_MINMAX_HPP
#define GPU_ALGORITHMS_MINMAX_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class T>
	GPU_DEVICE GPU_CONSTEXPR const T& max(const T& a, const T& b);

	template <class T, class Compare>
	GPU_DEVICE GPU_CONSTEXPR const T& max(const T& a, const T& b, Compare comp);

	template <class T>
	GPU_DEVICE GPU_CONSTEXPR const T& min(const T& a, const T& b);

	template <class T, class Compare>
	GPU_DEVICE GPU_CONSTEXPR const T& min(const T& a, const T& b, Compare comp);
}

#include <gstl/algorithms/minmax.cu>

#endif // GPU_ALGORITHMS_MINMAX_HPP
