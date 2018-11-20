#pragma once

#ifndef GPU_UTILITY_LIMITS_HPP
#define GPU_UTILITY_LIMITS_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	namespace detail
	{
		template <typename T, int digits, bool is_signed>
		struct compute_min
		{
			static const T value = T(T(1) << digits);
		};

		template <typename T, int digits>
		struct compute_min<T, digits, false>
		{
			static const T value = T(0);
		};
	}

	template <class T>
	class numeric_limits
	{
		public:
			using type = T;
			static const bool is_specialized = false;

			static const bool is_signed = type(-1) < type(0);
			static const int digits = static_cast<int>(sizeof(type) * 8 - is_signed);
			static const int digits10 = digits * 3 / 10;
			static const int max_digits10 = 0;
			static const type __min = detail::compute_min<type, digits, is_signed>::value;
			static const type __max = is_signed ? type(type(~0) ^ __min) : type(~0);
			GPU_DEVICE inline static type min() {return __min;}
			GPU_DEVICE inline static type max() {return __max;}
			GPU_DEVICE inline static type lowest() {return min();}

			static const bool is_integer = true;
			static const bool is_exact = true;
			static const int radix = 2;
			GPU_DEVICE inline static type epsilon() {return type(0);}
			GPU_DEVICE inline static type round_error() {return type(0);}

			static const int min_exponent = 0;
			static const int min_exponent10 = 0;
			static const int max_exponent = 0;
			static const int max_exponent10 = 0;

			static const bool has_infinity = false;
			static const bool has_quiet_NaN = false;
			static const bool has_signaling_NaN = false;
			static const bool has_denorm_loss = false;
			GPU_DEVICE inline static type infinity() {return type(0);}
			GPU_DEVICE inline static type quiet_NaN() {return type(0);}
			GPU_DEVICE inline static type signaling_NaN() {return type(0);}
			GPU_DEVICE inline static type denorm_min() {return type(0);}

			static const bool is_iec559 = false;
			static const bool is_bounded = true;
	};
}

#endif // GPU_UTILITY_LIMITS_HPP
