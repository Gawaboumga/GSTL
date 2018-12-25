#pragma once

#ifndef GPU_MEMORY_ALLOCATOR_TRAITS_HPP
#define GPU_MEMORY_ALLOCATOR_TRAITS_HPP

#include <gstl/prerequisites.hpp>

#include <memory>

namespace gpu
{
	template <class Allocator>
	struct allocator_traits
	{
		using allocator_type = Allocator;
		using value_type = typename std::allocator_traits<Allocator>::value_type;
		using pointer = typename std::allocator_traits<Allocator>::pointer;
		using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;
		using void_pointer = typename std::allocator_traits<Allocator>::void_pointer;
		using const_void_pointer = typename std::allocator_traits<Allocator>::const_void_pointer;
		using difference_type = typename std::allocator_traits<Allocator>::difference_type;
		using size_type = typename std::allocator_traits<Allocator>::size_type;

		GPU_DEVICE static pointer allocate(Allocator& a, size_type n);
		template <class Thread>
		GPU_DEVICE static pointer allocate(Thread g, Allocator& a, size_type n);
		GPU_DEVICE static pointer allocate(Allocator& a, size_type n, const_void_pointer hint);
		template <class Thread>
		GPU_DEVICE static pointer allocate(Thread g, Allocator& a, size_type n, const_void_pointer hint);

		template <class T, class... Args>
		GPU_DEVICE static void construct(Allocator& a, T* p, Args&&... args);
		template <class Thread, class T, class... Args>
		GPU_DEVICE static void construct(Thread g, Allocator& a, T* p, Args&&... args);
		template <class T, class... Args>
		GPU_DEVICE static void construct_n(Allocator& a, T* p, size_type n, Args&&... args);
		template <class Thread, class T, class... Args>
		GPU_DEVICE static void construct_n(Thread g, Allocator& a, T* p, size_type n, Args&&... args);
		template <class ForwardIt, class T, class... Args>
		GPU_DEVICE static void construct_range(Allocator& a, T* p, ForwardIt first, ForwardIt last);
		template <class Thread, class ForwardIt, class T, class... Args>
		GPU_DEVICE static void construct_range(Thread g, Allocator& a, T* p, ForwardIt first, ForwardIt last);

		GPU_DEVICE static void deallocate(Allocator& a, pointer p, size_type n);
		template <class T>
		GPU_DEVICE static void destroy(Allocator& a, T* p);

		GPU_DEVICE static size_type max_size(const Allocator& a);
	};
}

#include <gstl/memory/allocator_traits.cu>

#endif // GPU_MEMORY_ALLOCATOR_TRAITS_HPP
