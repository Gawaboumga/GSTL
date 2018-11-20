#include <gstl/math/intrinsics.cuh>

namespace gpu
{
	namespace detail
	{
		template <typename T>
		struct brev
		{
		};

		template <>
		struct brev<unsigned int>
		{
			GPU_DEVICE unsigned int operator()(unsigned int x) const
			{
				return __brev(x);
			}
		};

		template <>
		struct brev<unsigned long long>
		{
			GPU_DEVICE unsigned long long operator()(unsigned long long x) const
			{
				return __brevll(x);
			}
		};

		template <typename T>
		struct clz
		{
		};

		template <>
		struct clz<unsigned int>
		{
			GPU_DEVICE unsigned int operator()(unsigned int x) const
			{
				return __clz(x);
			}
		};

		template <>
		struct clz<unsigned long long>
		{
			GPU_DEVICE unsigned long long operator()(unsigned long long x) const
			{
				return __clzll(x);
			}
		};

		template <typename T>
		struct ffs
		{
		};

		template <>
		struct ffs<unsigned int>
		{
			GPU_DEVICE unsigned int operator()(unsigned int x) const
			{
				return __ffs(x);
			}
		};

		template <>
		struct ffs<unsigned long long>
		{
			GPU_DEVICE unsigned long long operator()(unsigned long long x) const
			{
				return __ffsll(x);
			}
		};

		template <typename T>
		struct popc
		{
		};

		template <>
		struct popc<unsigned int>
		{
			GPU_DEVICE unsigned int operator()(unsigned int x) const
			{
				return __popc(x);
			}
		};

		template <>
		struct popc<unsigned long long>
		{
			GPU_DEVICE unsigned long long operator()(unsigned long long x) const
			{
				return __popcll(x);
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
