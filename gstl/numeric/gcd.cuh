#pragma once

#ifndef GPU_NUMERIC_GCD_HPP
#define GPU_NUMERIC_GCD_HPP

#include <type_traits>

namespace gpu
{
	template <class M, class N>
	GPU_DEVICE GPU_CONSTEXPR std::common_type_t<M, N> gcd(M m, N n);

	template <class M, class N>
	GPU_DEVICE GPU_CONSTEXPR std::common_type_t<M, N> lcm(M m, N n);
}

#include <gstl/numeric/gcd.cu>

#endif // GPU_NUMERIC_GCD_HPP
