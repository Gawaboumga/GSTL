#include <gstl/numeric/gcd.cuh>

namespace gpu
{
	template <class M, class N>
	GPU_DEVICE GPU_CONSTEXPR std::common_type_t<M, N> gcd(M m, N n)
	{
		do
		{
			M t = m % n;
			m = n;
			n = t;
		} while (n);
		return m;
	}

	template <class M, class N>
	GPU_DEVICE GPU_CONSTEXPR std::common_type_t<M, N> lcm(M m, N n)
	{
		if (m == 0 || n == 0)
			return 0;

		auto smallest_value = m / gcd(m, n);
		return smallest_value * n;
	}
}
