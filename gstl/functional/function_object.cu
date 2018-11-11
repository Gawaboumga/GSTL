#include <gstl/functional/function_object.cuh>

#include <utility>

namespace gpu
{
	// shamelessly stolen from: https://stackoverflow.com/questions/15202474/default-function-that-just-returns-the-passed-value
	// by user Mankarse on Mar 4 2013 at 13:25
	struct identity
	{
		template<typename U>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(U&& v) const noexcept -> decltype(std::forward<U>(v))
		{
			return std::forward<U>(v);
		}
	};

	template <>
	struct plus<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) + static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) + static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct minus<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) - static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) - static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct multiplies<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) * static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) * static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct divides<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) / static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) / static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct modulus<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) % static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) % static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct negate<void>
	{
		template <class T>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T&& arg) const -> decltype(-static_cast<T&&>(arg))
		{
			return (-arg);
		}
	};

	template <>
	struct equal_to<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) == static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) == static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct not_equal_to<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) != static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) != static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct greater<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) > static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) > static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct less<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) < static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) < static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct greater_equal<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) >= static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) >= static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct less_equal<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) <= static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) <= static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct logical_and<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) && static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) && static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct logical_or<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) || static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) || static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct logical_not<void>
	{
		template <class T>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T&& arg) const -> decltype(!static_cast<T&&>(arg))
		{
			return (!arg);
		}
	};

	template <>
	struct bit_and<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) & static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) & static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct bit_or<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) | static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) | static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct bit_xor<void>
	{
		template <class T1, class T2>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T1&& lhs, T2&& rhs) const -> decltype(static_cast<T1&&>(lhs) ^ static_cast<T2&&>(rhs))
		{
			return (static_cast<T1&&>(lhs) ^ static_cast<T2&&>(rhs));
		}
	};

	template <>
	struct bit_not<void>
	{
		template <class T>
		GPU_DEVICE GPU_CONSTEXPR auto operator()(T&& arg) const -> decltype(~static_cast<T&&>(arg))
		{
			return (~arg);
		}
	};
}
