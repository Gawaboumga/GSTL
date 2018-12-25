#pragma once

#ifndef GPU_MEMORY_ALGORITHMS_HPP
#define GPU_MEMORY_ALGORITHMS_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class InputIt, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_copy(InputIt first, InputIt last, ForwardIt d_first);

	template <class Thread, class InputIt, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_copy(Thread g, InputIt first, InputIt last, ForwardIt d_first);

	template <class InputIt, class Size, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_copy_n(InputIt first, Size count, ForwardIt d_first);

	template <class Thread, class InputIt, class Size, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_copy_n(Thread g, InputIt first, Size count, ForwardIt d_first);

	template <class ForwardIt, typename T>
	GPU_DEVICE void uninitialized_fill(ForwardIt first, ForwardIt last, const T& value);

	template <class Thread, class ForwardIt, typename T>
	GPU_DEVICE void uninitialized_fill(Thread g, ForwardIt first, ForwardIt last, const T& value);

	template <class ForwardIt, class Size, typename T>
	GPU_DEVICE ForwardIt uninitialized_fill_n(ForwardIt first, Size count, const T& value);

	template <class Thread, class ForwardIt, class Size, typename T>
	GPU_DEVICE ForwardIt uninitialized_fill_n(Thread g, ForwardIt first, Size count, const T& value);

	template <class InputIt, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_move(InputIt first, InputIt last, ForwardIt d_first);

	template <class Thread, class InputIt, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_move(Thread g, InputIt first, InputIt last, ForwardIt d_first);

	template <class InputIt, class Size, class ForwardIt>
	GPU_DEVICE std::pair<InputIt, ForwardIt> uninitialized_move_n(InputIt first, Size count, ForwardIt d_first);
	
	template <class Thread, class InputIt, class Size, class ForwardIt>
	GPU_DEVICE std::pair<InputIt, ForwardIt> uninitialized_move_n(Thread g, InputIt first, Size count, ForwardIt d_first);

	template <class ForwardIt>
	GPU_DEVICE void uninitialized_default_construct(ForwardIt first, ForwardIt last);

	template <class Thread, class ForwardIt>
	GPU_DEVICE void uninitialized_default_construct(Thread g, ForwardIt first, ForwardIt last);

	template <class ForwardIt, class Size>
	GPU_DEVICE ForwardIt uninitialized_default_construct_n(ForwardIt first, Size n);

	template <class Thread, class ForwardIt, class Size>
	GPU_DEVICE ForwardIt uninitialized_default_construct_n(Thread g, ForwardIt first, Size n);

	template <class ForwardIt>
	GPU_DEVICE void uninitialized_value_construct(ForwardIt first, ForwardIt last);

	template <class Thread, class ForwardIt>
	GPU_DEVICE void uninitialized_value_construct(Thread g, ForwardIt first, ForwardIt last);

	template <class ForwardIt, class Size>
	GPU_DEVICE ForwardIt uninitialized_value_construct_n(ForwardIt first, Size n);

	template <class Thread, class ForwardIt, class Size>
	GPU_DEVICE ForwardIt uninitialized_value_construct_n(Thread g, ForwardIt first, Size n);

	template <typename T>
	GPU_DEVICE void destroy_at(T* p);

	template <class ForwardIt>
	GPU_DEVICE void destroy(ForwardIt first, ForwardIt last);

	template <class Thread, class ForwardIt>
	GPU_DEVICE void destroy(Thread g, ForwardIt first, ForwardIt last);

	template <class ForwardIt, class Size>
	GPU_DEVICE ForwardIt destroy_n(ForwardIt first, Size n);
	
	template <class Thread, class ForwardIt, class Size>
	GPU_DEVICE ForwardIt destroy_n(Thread g, ForwardIt first, Size n);
}

#include <gstl/memory/algorithms.cu>

#endif // GPU_MEMORY_ALGORITHMS_HPP
