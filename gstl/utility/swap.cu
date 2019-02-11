#include <gstl/utility/swap.cuh>

namespace gpu
{
	template <class ForwardIt1, class ForwardIt2>
	GPU_DEVICE GPU_CONSTEXPR void iter_swap(ForwardIt1 a, ForwardIt2 b)
	{
		using gpu::swap;
		swap(*a, *b);
	}

	template <typename T>
	GPU_DEVICE GPU_CONSTEXPR void swap(T& a, T& b) noexcept
	{
		T tmp = std::move(b);
		b = std::move(a);
		a = std::move(tmp);
	}

	template <typename T, std::size_t N>
	GPU_DEVICE GPU_CONSTEXPR void swap(T (&a)[N], T (&b)[N]) noexcept
	{
		swap_ranges(a, a + N, b);
	}
}
