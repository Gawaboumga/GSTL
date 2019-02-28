#include <gstl/math/bit.cuh>

#include <gstl/math/intrinsics.cuh>
#include <gstl/utility/limits.cuh>

namespace gpu
{
	template <typename To, typename From>
	GPU_DEVICE GPU_CONSTEXPR To bit_cast(const From& from) noexcept
	{
		return reinterpret_cast<To>(from);
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T ceil2(T x) noexcept
	{
		if (x == 0 || x == 1)
			return T(1);

		return 1 << (32u - clz(x - 1u));
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T floor2(T x) noexcept
	{
		if (x == 0)
			return 0;

		constexpr auto number_of_digits = numeric_limits<T>::digits;
		return 1u << (number_of_digits - clz(x));
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR bool ispow2(T x) noexcept
	{
		return popc(x) == 1;
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR T log2p1(T x) noexcept
	{
		constexpr auto number_of_digits = numeric_limits<T>::digits;
		return number_of_digits - clz(x);
	}
}
