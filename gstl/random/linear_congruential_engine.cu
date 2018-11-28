#include <gstl/random/linear_congruential_engine.cuh>

namespace gpu
{
	namespace detail
	{
		template <unsigned long long a, unsigned long long c, unsigned long long m, unsigned long long Mp, bool MightOverflow = (a != 0 && m != 0 && m - 1 > (Mp - c) / a)>
		struct lce_ta;

		// 64

		template <unsigned long long a, unsigned long long c, unsigned long long m>
		struct lce_ta<a, c, m, (unsigned long long)(~0), true>
		{
			using result_type = unsigned long long;
			GPU_DEVICE static result_type next(result_type x)
			{
				// Schrage's algorithm
				const result_type q = m / a;
				const result_type r = m % a;
				const result_type t0 = a * (x % q);
				const result_type t1 = r * (x / q);
				x = t0 + (t0 < t1) * m - t1;
				x += c - (x >= m - c) * m;
				return x;
			}
		};

		template <unsigned long long a, unsigned long long m>
		struct lce_ta<a, 0, m, (unsigned long long)(~0), true>
		{
			using result_type = unsigned long long;
			GPU_DEVICE static result_type next(result_type x)
			{
				// Schrage's algorithm
				const result_type q = m / a;
				const result_type r = m % a;
				const result_type t0 = a * (x % q);
				const result_type t1 = r * (x / q);
				x = t0 + (t0 < t1) * m - t1;
				return x;
			}
		};

		template <unsigned long long a, unsigned long long c, unsigned long long m>
		struct lce_ta<a, c, m, (unsigned long long)(~0), false>
		{
			using result_type = unsigned long long;
			GPU_DEVICE static result_type next(result_type x)
			{
				return (a * x + c) % m;
			}
		};

		template <unsigned long long a, unsigned long long c>
		struct lce_ta<a, c, 0, (unsigned long long)(~0), false>
		{
			using result_type = unsigned long long;
			GPU_DEVICE static result_type next(result_type x)
			{
				return a * x + c;
			}
		};

		// 32

		template <unsigned long long Ap, unsigned long long Cp, unsigned long long Mp>
		struct lce_ta<Ap, Cp, Mp, unsigned(~0), true>
		{
			using result_type = unsigned int;
			GPU_DEVICE static result_type next(result_type x)
			{
				const result_type a = static_cast<result_type>(Ap);
				const result_type c = static_cast<result_type>(Cp);
				const result_type m = static_cast<result_type>(Mp);
				// Schrage's algorithm
				const result_type q = m / a;
				const result_type r = m % a;
				const result_type t0 = a * (x % q);
				const result_type t1 = r * (x / q);
				x = t0 + (t0 < t1) * m - t1;
				x += c - (x >= m - c) * m;
				return x;
			}
		};

		template <unsigned long long Ap, unsigned long long Mp>
		struct lce_ta<Ap, 0, Mp, unsigned(~0), true>
		{
			using result_type = unsigned int;
			GPU_DEVICE static result_type next(result_type x)
			{
				const result_type a = static_cast<result_type>(Ap);
				const result_type m = static_cast<result_type>(Mp);
				// Schrage's algorithm
				const result_type q = m / a;
				const result_type r = m % a;
				const result_type t0 = a * (x % q);
				const result_type t1 = r * (x / q);
				x = t0 + (t0 < t1) * m - t1;
				return x;
			}
		};

		template <unsigned long long Ap, unsigned long long Cp, unsigned long long Mp>
		struct lce_ta<Ap, Cp, Mp, unsigned(~0), false>
		{
			using result_type = unsigned int; 
			GPU_DEVICE static result_type next(result_type x)
			{
				const result_type a = static_cast<result_type>(Ap);
				const result_type c = static_cast<result_type>(Cp);
				const result_type m = static_cast<result_type>(Mp);
				return (a * x + c) % m;
			}
		};

		template <unsigned long long Ap, unsigned long long Cp>
		struct lce_ta<Ap, Cp, 0, unsigned(~0), false>
		{
			using result_type = unsigned int; 
			GPU_DEVICE static result_type next(result_type x)
			{
				const result_type a = static_cast<result_type>(Ap);
				const result_type c = static_cast<result_type>(Cp);
				return a * x + c;
			}
		};

		// 16

		template <unsigned long long a, unsigned long long c, unsigned long long m, bool b>
		struct lce_ta<a, c, m, (unsigned short)(~0), b>
		{
			using result_type = unsigned short;
			GPU_DEVICE static result_type next(result_type x)
			{
				return static_cast<result_type>(lce_ta<a, c, m, unsigned(~0)>::next(x));
			}
		};
	}

	template <class UIntType, UIntType a, UIntType c, UIntType m>
	GPU_DEVICE linear_congruential_engine<UIntType, a, c, m>::linear_congruential_engine(result_type s)
	{
		seed(s);
	}

	template <class UIntType, UIntType a, UIntType c, UIntType m>
	GPU_DEVICE void linear_congruential_engine<UIntType, a, c, m>::discard(unsigned long long z)
	{
		for (; z; --z)
			operator()();
	}

	template <class UIntType, UIntType a, UIntType c, UIntType m>
	GPU_DEVICE typename linear_congruential_engine<UIntType, a, c, m>::result_type linear_congruential_engine<UIntType, a, c, m>::operator()()
	{
		return m_x = static_cast<result_type>(detail::lce_ta<a, c, m, Mp>::next(m_x));
	}

	template <class UIntType, UIntType a, UIntType c, UIntType m>
	GPU_DEVICE void linear_congruential_engine<UIntType, a, c, m>::seed(result_type s)
	{
		m_x = s;
	}

	template <class UIntType, UIntType a, UIntType c, UIntType m>
	GPU_DEVICE GPU_CONSTEXPR typename linear_congruential_engine<UIntType, a, c, m>::result_type linear_congruential_engine<UIntType, a, c, m>::max()
	{
		return m - 1u;
	}

	template <class UIntType, UIntType a, UIntType c, UIntType m>
	GPU_DEVICE GPU_CONSTEXPR typename linear_congruential_engine<UIntType, a, c, m>::result_type linear_congruential_engine<UIntType, a, c, m>::min()
	{
		return c == 0u ? 1u : 0u;
	}

	template <class UIntType, UIntType a, UIntType c, UIntType m>
	GPU_DEVICE bool operator==(const linear_congruential_engine<UIntType, a, c, m>& x, const linear_congruential_engine<UIntType, a, c, m>& y)
	{
		return x.m_x == y.m_x;
	}

	template <class UIntType, UIntType a, UIntType c, UIntType m>
	GPU_DEVICE bool operator!=(const linear_congruential_engine<UIntType, a, c, m>& x, const linear_congruential_engine<UIntType, a, c, m>& y)
	{
		return !(x == y);
	}
}
