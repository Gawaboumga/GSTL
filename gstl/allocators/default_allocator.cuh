#pragma once

#ifndef GPU_ALLOCATORS_DEFAULT_ALLOCATOR_HPP
#define GPU_ALLOCATORS_DEFAULT_ALLOCATOR_HPP

#include <gstl/prerequisites.hpp>

#include <gstl/allocators/linear_allocator.cuh>

namespace gpu
{
	template <class T>
	using allocator = linear_allocator<T>;
}

#endif // GPU_ALLOCATORS_DEFAULT_ALLOCATOR_HPP
