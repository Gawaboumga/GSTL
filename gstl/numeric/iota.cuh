#pragma once

#ifndef GPU_NUMERIC_IOTA_HPP
#define GPU_NUMERIC_IOTA_HPP

namespace gpu
{
	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR void iota(InputIt first, InputIt last, T value);
}

#include <gstl/numeric/iota.cu>

#endif // GPU_NUMERIC_IOTA_HPP
