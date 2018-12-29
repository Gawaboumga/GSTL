#pragma once

#ifndef GPU_MATH_BIT_HPP
#define GPU_MATH_BIT_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <typename To, typename From>
	GPU_DEVICE GPU_CONSTEXPR To bit_cast(const From& from) noexcept;

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T ceil2(T x) noexcept;

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T floor2(T x) noexcept;

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR bool ispow2(T x) noexcept;

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T log2p1(T x) noexcept;
}

#include <gstl/math/bit.cu>

#endif // GPU_MATH_BIT_HPP
