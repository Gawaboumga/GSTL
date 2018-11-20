#pragma once

#ifndef GPU_MATH_INTRINSICS_HPP
#define GPU_MATH_INTRINSICS_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <typename T>
	GPU_DEVICE T brev(T x);

	template <typename T>
	GPU_DEVICE T clz(T x);

	template <typename T>
	GPU_DEVICE T ffs(T x);

	template <typename T>
	GPU_DEVICE T popc(T x);
}

#include <gstl/math/intrinsics.cu>

#endif // GPU_MATH_INTRINSICS_HPP
