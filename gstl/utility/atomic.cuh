#pragma once

#ifndef GPU_UTILITY_ATOMIC_HPP
#define GPU_UTILITY_ATOMIC_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <typename T>
	struct atomic
	{
		using value_type = T;
		using difference_type = value_type;
		using threads = block_tile_t<32>;

		atomic() = default;
		GPU_DEVICE atomic(value_type desired);

		GPU_DEVICE value_type compare_and_swap(value_type expected, value_type desired);
		GPU_DEVICE value_type compare_and_swap(threads g, value_type expected, value_type desired);
		GPU_DEVICE bool compare_exchange_strong(value_type expected, value_type desired);
		GPU_DEVICE bool compare_exchange_weak(value_type expected, value_type desired);

		GPU_DEVICE const value_type* data() const;
		GPU_DEVICE value_type decrement(value_type min);

		GPU_DEVICE value_type exchange(value_type desired);

		GPU_DEVICE value_type fetch_add(value_type arg);
		GPU_DEVICE value_type fetch_and(value_type arg);
		GPU_DEVICE value_type fetch_or(value_type arg);
		GPU_DEVICE value_type fetch_sub(value_type arg);
		GPU_DEVICE value_type fetch_xor(value_type arg);

		GPU_DEVICE value_type increment(value_type max);
		GPU_DEVICE bool is_lock_free() const;

		GPU_DEVICE value_type load() const;

		GPU_DEVICE value_type max(value_type arg);
		GPU_DEVICE value_type max(threads g, value_type arg);
		GPU_DEVICE value_type min(value_type arg);
		GPU_DEVICE value_type min(threads g, value_type arg);

		GPU_DEVICE operator value_type() const;
		GPU_DEVICE value_type operator=(value_type desired);
		GPU_DEVICE value_type operator++();
		GPU_DEVICE value_type operator++(int);
		GPU_DEVICE value_type operator--();
		GPU_DEVICE value_type operator--(int);
		GPU_DEVICE value_type operator+=(value_type arg);
		GPU_DEVICE value_type operator-=(value_type arg);
		GPU_DEVICE value_type operator&=(value_type arg);
		GPU_DEVICE value_type operator|=(value_type arg);
		GPU_DEVICE value_type operator^=(value_type arg);

		GPU_DEVICE void store(value_type desired);
		GPU_DEVICE void store(threads g, value_type desired);
		GPU_DEVICE void store_unatomically(value_type desired);

		private:
			value_type m_data;
	};

	template <typename T>
	struct atomic<T*>
	{
		using pointer_type = T*;
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using threads = cooperative_groups::thread_block_tile<32>;

		atomic() = default;
		GPU_DEVICE atomic(pointer_type desired);

		GPU_DEVICE pointer_type compare_and_swap(pointer_type expected, pointer_type desired);
		GPU_DEVICE pointer_type compare_and_swap(threads group, pointer_type expected, pointer_type desired);
		GPU_DEVICE bool compare_exchange_strong(pointer_type expected, pointer_type desired);
		GPU_DEVICE bool compare_exchange_weak(pointer_type expected, pointer_type desired);

		GPU_DEVICE pointer_type exchange(pointer_type desired);

		GPU_DEVICE pointer_type fetch_add(difference_type arg);
		GPU_DEVICE pointer_type fetch_sub(difference_type arg);

		GPU_DEVICE bool is_lock_free() const;

		GPU_DEVICE pointer_type load() const;

		GPU_DEVICE operator pointer_type() const;
		GPU_DEVICE pointer_type operator=(pointer_type desired);
		GPU_DEVICE pointer_type operator++();
		GPU_DEVICE pointer_type operator++(int);
		GPU_DEVICE pointer_type operator--();
		GPU_DEVICE pointer_type operator--(int);
		GPU_DEVICE pointer_type operator+=(difference_type arg);
		GPU_DEVICE pointer_type operator-=(difference_type arg);

		GPU_DEVICE void store(pointer_type desired);
		GPU_DEVICE void store(threads group, pointer_type desired);
		GPU_DEVICE void store_unatomically(pointer_type desired);

		private:
			GPU_DEVICE std::uintptr_t* data();
			GPU_DEVICE std::uintptr_t to_integral(pointer_type value) const;
			GPU_DEVICE pointer_type to_pointer(std::uintptr_t value) const;

			pointer_type m_data;
	};
}

#include <gstl/utility/atomic.cu>

#endif // GPU_UTILITY_ATOMIC_HPP
