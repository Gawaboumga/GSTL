#include <gstl/utility/atomic.cuh>

#include <type_traits>

namespace gpu
{
	namespace detail
	{
		template <typename T, bool enum_like = std::is_enum<T>::value>
		struct atomic_cas
		{
		};

		template <typename T>
		struct atomic_cas<T, true>
		{
			GPU_DEVICE T operator()(T* address, T compare, T val) const noexcept
			{
				return static_cast<T>(atomic_cas<std::underlying_type_t<T>>{}(
					reinterpret_cast<std::underlying_type_t<T>*>(address),
					compare,
					val));
			}
		};

		template <>
		struct atomic_cas<int, false>
		{
			GPU_DEVICE int operator()(int* address, int compare, int val) const noexcept
			{
				return atomicCAS(address, compare, val);
			}
		};

		template <>
		struct atomic_cas<unsigned int, false>
		{
			GPU_DEVICE unsigned int operator()(unsigned int* address, unsigned int compare, unsigned int val) const noexcept
			{
				return atomicCAS(address, compare, val);
			}
		};

		template <>
		struct atomic_cas<unsigned long long int, false>
		{
			GPU_DEVICE unsigned long long int operator()(unsigned long long int* address, unsigned long long int compare, unsigned long long int val) const noexcept
			{
				return atomicCAS(address, compare, val);
			}
		};

		template <typename T, bool enum_like = std::is_enum<T>::value>
		struct atomic_exchange
		{
		};

		template <typename T>
		struct atomic_exchange<T, true>
		{
			GPU_DEVICE T operator()(T* address, T val) const noexcept
			{
				return static_cast<T>(atomic_exchange<std::underlying_type_t<T>>{}(
					reinterpret_cast<std::underlying_type_t<T>*>(address),
					val));
			}
		};

		template <>
		struct atomic_exchange<int, false>
		{
			GPU_DEVICE int operator()(int* address, int val) const noexcept
			{
				return atomicExch(address, val);
			}
		};

		template <>
		struct atomic_exchange<unsigned int, false>
		{
			GPU_DEVICE unsigned int operator()(unsigned int* address, unsigned int val) const noexcept
			{
				return atomicExch(address, val);
			}
		};

		template <>
		struct atomic_exchange<unsigned long long int, false>
		{
			GPU_DEVICE unsigned long long int operator()(unsigned long long int* address, unsigned long long int val) const noexcept
			{
				return atomicExch(address, val);
			}
		};

		template <>
		struct atomic_exchange<float, false>
		{
			GPU_DEVICE float operator()(float* address, float val) const noexcept
			{
				return atomicExch(address, val);
			}
		};
	}

	template <typename T>
	GPU_DEVICE atomic<T>::atomic(value_type desired) :
		m_data{ desired }
	{
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::compare_and_swap(value_type expected, value_type desired)
	{
		return detail::atomic_cas<T>{}(&m_data, expected, desired);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::compare_and_swap(threads g, value_type expected, value_type desired)
	{
		value_type old;
		if (g.thread_rank() == 0)
			old = detail::atomic_cas<T>{}(&m_data, expected, desired);
		old = g.shfl(old, 0);
		return old;
	}

	template <typename T>
	GPU_DEVICE bool atomic<T>::compare_exchange_strong(value_type expected, value_type desired)
	{
		value_type old;
		do
		{
			old = detail::atomic_cas<T>{}(&m_data, expected, desired);
		} while (expected != old);
		return true;
	}

	template <typename T>
	GPU_DEVICE bool atomic<T>::compare_exchange_weak(value_type expected, value_type desired)
	{
		return detail::atomic_cas<T>{}(&m_data, expected, desired) == desired;
	}

	template <typename T>
	GPU_DEVICE const typename atomic<T>::value_type* atomic<T>::data() const
	{
		return &m_data;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::decrement(value_type min)
	{
		return atomicDec(&m_data, min);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::exchange(value_type desired)
	{
		return detail::atomic_exchange<T>{}(&m_data, desired);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::fetch_add(value_type arg)
	{
		return atomicAdd(&m_data, arg);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::fetch_and(value_type arg)
	{
		return atomicAnd(&m_data, arg);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::fetch_or(value_type arg)
	{
		return atomicOr(&m_data, arg);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::fetch_sub(value_type arg)
	{
		return atomicSub(&m_data, arg);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::fetch_xor(value_type arg)
	{
		return atomicXor(&m_data, arg);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::increment(value_type max)
	{
		return atomicInc(&m_data, max);
	}

	template <typename T>
	GPU_DEVICE bool atomic<T>::is_lock_free() const
	{
		return true;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::load() const
	{
	#ifdef GPU_DEBUG_ATOMIC
		ENSURE(reinterpret_cast<uintptr_t>(&m_data) > 0x1000);
	#endif // GPU_DEBUG_ATOMIC
		return m_data;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::max(value_type arg)
	{
		return atomicMax(&m_data, arg);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::max(threads g, value_type arg)
	{
		value_type old;
		if (g.thread_rank() == 0)
			old = atomicMax(&m_data, arg);
		old = g.shfl(old, 0);
		return old;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::min(value_type arg)
	{
		return atomicMin(&m_data, arg);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::min(threads g, value_type arg)
	{
		value_type old;
		if (g.thread_rank() == 0)
			old = atomicMin(&m_data, arg);
		old = g.shfl(old, 0);
		return old;
	}

	template <typename T>
	GPU_DEVICE atomic<T>::operator value_type() const
	{
		return load();
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator=(value_type desired)
	{
		return atomicExch(&m_data, desired);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator++()
	{
		return fetch_add(value_type{ 1 }) + value_type{ 1 };
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator++(int)
	{
		return fetch_add(value_type{ 1 });
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator--()
	{
		return fetch_sub(value_type{ 1 }) - value_type{ 1 };
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator--(int)
	{
		return fetch_sub(value_type{ 1 });
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator+=(value_type arg)
	{
		return fetch_add(arg) + arg;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator-=(value_type arg)
	{
		return fetch_sub(arg) - arg;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator&=(value_type arg)
	{
		return fetch_and(arg) + arg;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator|=(value_type arg)
	{
		return fetch_or(arg) + arg;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T>::value_type atomic<T>::operator^=(value_type arg)
	{
		return fetch_xor(arg) + arg;
	}

	template <typename T>
	GPU_DEVICE void atomic<T>::store(value_type desired)
	{
		detail::atomic_exchange<T>{}(&m_data, desired);
	}

	template <typename T>
	GPU_DEVICE void atomic<T>::store(threads group, value_type desired)
	{
		if (group.thread_rank() == 0)
			store(desired);
	}

	template <typename T>
	GPU_DEVICE void atomic<T>::store_unatomically(value_type desired)
	{
		m_data = desired;
	}

	template <typename T>
	GPU_DEVICE atomic<T*>::atomic(pointer_type desired) :
		m_data{ desired }
	{
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::compare_and_swap(pointer_type expected, pointer_type desired)
	{
		return to_pointer(detail::atomic_cas<T>{}(data(), to_integral(expected), to_integral(desired)));
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::compare_and_swap(threads group, pointer_type expected, pointer_type desired)
	{
		std::uintptr_t result;
		if (group.thread_rank() == 0)
			result = to_integral(compare_and_swap(expected, desired));
		return to_pointer(group.shfl(result, 0));
	}

	template <typename T>
	GPU_DEVICE bool atomic<T*>::compare_exchange_strong(pointer_type expected, pointer_type desired)
	{
		pointer_type old;
		do
		{
			old = compare_and_swap(expected, desired);
		} while (expected != old);
		return true;
	}

	template <typename T>
	GPU_DEVICE bool atomic<T*>::compare_exchange_weak(pointer_type expected, pointer_type desired)
	{
		return compare_and_swap(expected, desired) == to_integral(desired);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::exchange(pointer_type desired)
	{
		return to_pointer(detail::atomic_exchange<T>{}(data(), to_integral(desired)));
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::fetch_add(difference_type arg)
	{
		return to_pointer(atomicAdd(data(), arg));
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::fetch_sub(difference_type arg)
	{
		return to_pointer(atomicSub(data(), arg));
	}

	template <typename T>
	GPU_DEVICE bool atomic<T*>::is_lock_free() const
	{
		return true;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::load() const
	{
		return m_data;
	}

	template <typename T>
	GPU_DEVICE atomic<T*>::operator pointer_type() const
	{
		return load();
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::operator=(pointer_type desired)
	{
		return to_pointer(detail::atomic_exchange<T>{}(data(), to_integral(desired)));
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::operator++()
	{
		return fetch_add(difference_type{ 1 }) + difference_type{ 1 };
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::operator++(int)
	{
		return fetch_add(difference_type{ 1 });
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::operator--()
	{
		return fetch_sub(difference_type{ 1 }) - difference_type{ 1 };
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::operator--(int)
	{
		return fetch_sub(difference_type{ 1 });
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::operator+=(difference_type arg)
	{
		return fetch_add(arg) + arg;
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::operator-=(difference_type arg)
	{
		return fetch_sub(arg) - arg;
	}

	template <typename T>
	GPU_DEVICE void atomic<T*>::store(pointer_type desired)
	{
		detail::atomic_exchange<T>{}(data(), to_integral(desired));
	}

	template <typename T>
	GPU_DEVICE void atomic<T*>::store(threads group, pointer_type desired)
	{
		if (group.thread_rank() == 0)
			store(desired);
	}

	template <typename T>
	GPU_DEVICE void atomic<T*>::store_unatomically(pointer_type desired)
	{
		m_data = desired;
	}

	template <typename T>
	GPU_DEVICE std::uintptr_t* atomic<T*>::data()
	{
	#ifdef GPU_DEBUG_ATOMIC
		ENSURE(reinterpret_cast<uintptr_t>(&m_data) > 0x1000);
	#endif // GPU_DEBUG_ATOMIC
		return reinterpret_cast<std::uintptr_t*>(&m_data);
	}

	template <typename T>
	GPU_DEVICE std::uintptr_t atomic<T*>::to_integral(pointer_type value) const
	{
		return reinterpret_cast<std::uintptr_t>(value);
	}

	template <typename T>
	GPU_DEVICE typename atomic<T*>::pointer_type atomic<T*>::to_pointer(std::uintptr_t value) const
	{
		return reinterpret_cast<pointer_type>(value);
	}
}
