#include <gstl/math/intrinsics.cuh>

namespace gpu
{
	namespace detail
	{
		template <typename T, bool arithmetic = std::is_arithmetic<T>::value, bool four_bytes_or_less = sizeof(T) <= 4>
		struct brev
		{
		};

		template <typename T>
		struct brev<T, true, true>
		{
			GPU_DEVICE T operator()(T x) const
			{
				return static_cast<T>(__brev(static_cast<unsigned int>(x)));
			}
		};

		template <typename T>
		struct brev<T, true, false>
		{
			GPU_DEVICE T operator()(T x) const
			{
				return static_cast<T>(__brevll(static_cast<unsigned long long int>(x)));
			}
		};

		template <typename T, bool arithmetic = std::is_arithmetic<T>::value, bool four_bytes_or_less = sizeof(T) <= 4>
		struct clz
		{
		};

		template <typename T>
		struct clz<T, true, true>
		{
			GPU_DEVICE T operator()(T x) const
			{
				return static_cast<T>(__clz(static_cast<unsigned int>(x)));
			}
		};

		template <typename T>
		struct clz<T, true, false>
		{
			GPU_DEVICE T operator()(T x) const
			{
				return static_cast<T>(__clzll(static_cast<unsigned long long int>(x)));
			}
		};

		template <typename T, bool arithmetic = std::is_arithmetic<T>::value, bool four_bytes_or_less = sizeof(T) <= 4>
		struct ffs
		{
		};

		template <typename T>
		struct ffs<T, true, true>
		{
			GPU_DEVICE T operator()(T x) const
			{
				return static_cast<T>(__ffs(static_cast<unsigned int>(x)));
			}
		};

		template <typename T>
		struct ffs<T, true, false>
		{
			GPU_DEVICE T operator()(T x) const
			{
				return static_cast<T>(__ffsll(static_cast<unsigned long long int>(x)));
			}
		};

		template <typename T, bool arithmetic = std::is_arithmetic<T>::value, bool four_bytes_or_less = sizeof(T) <= 4>
		struct popc
		{
		};

		template <typename T>
		struct popc<T, true, true>
		{
			GPU_DEVICE T operator()(T x) const
			{
				return static_cast<T>(__popc(static_cast<unsigned int>(x)));
			}
		};

		template <typename T>
		struct popc<T, true, false>
		{
			GPU_DEVICE T operator()(T x) const
			{
				return static_cast<T>(__popcll(static_cast<unsigned long long int>(x)));
			}
		};
	}

	template <typename T>
	GPU_DEVICE T brev(T x)
	{
		return detail::brev<T>()(x);
	}

	template <typename T>
	GPU_DEVICE T clz(T x)
	{
		return detail::clz<T>()(x);
	}

	template <typename T>
	GPU_DEVICE T ffs(T x)
	{
		return detail::ffs<T>()(x);
	}

	template <typename T>
	GPU_DEVICE T popc(T x)
	{
		return detail::popc<T>()(x);
	}
}
