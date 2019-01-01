#include <gstl/algorithms/minmax.cuh>

namespace gpu
{
	template <class T>
	GPU_DEVICE GPU_CONSTEXPR const T& max(const T& a, const T& b)
	{
		return (a < b) ? b : a;
	}

	template <class T, class Compare>
	GPU_DEVICE GPU_CONSTEXPR const T& max(const T& a, const T& b, Compare comp)
	{
		return (comp(a, b)) ? b : a;
	}

	template <class T>
	GPU_DEVICE GPU_CONSTEXPR const T& min(const T& a, const T& b)
	{
		return (b < a) ? b : a;
	}

	template <class T, class Compare>
	GPU_DEVICE GPU_CONSTEXPR const T& min(const T& a, const T& b, Compare comp)
	{
		return (comp(b, a)) ? b : a;
	}
}
