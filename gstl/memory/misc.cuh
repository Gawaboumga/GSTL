#pragma once

#ifndef GPU_MEMORY_MISC_HPP
#define GPU_MEMORY_MISC_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	inline GPU_DEVICE void* align(size_t alignment, size_t size, void*& ptr, size_t& space);

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T* addressof(T& arg) noexcept;

	template <typename T>
	GPU_DEVICE const T* addressof(const T&&) = delete;
}

#include <gstl/memory/misc.cu>

#endif // GPU_MEMORY_MISC_HPP
