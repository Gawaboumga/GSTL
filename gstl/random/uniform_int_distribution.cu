#include <gstl/random/uniform_int_distribution.cuh>

namespace gpu
{
	template <class IntType>
	GPU_DEVICE uniform_int_distribution<IntType>::param_type::param_type(IntType a, IntType b) :
		m_a(a),
		m_b(b)
	{
	}

	template <class IntType>
	GPU_DEVICE typename uniform_int_distribution<IntType>::result_type uniform_int_distribution<IntType>::param_type::a() const
	{
		return m_a;
	}

	template <class IntType>
	GPU_DEVICE typename uniform_int_distribution<IntType>::result_type uniform_int_distribution<IntType>::param_type::b() const
	{
		return m_b;
	}

	template <class IntType>
	GPU_DEVICE bool operator==(const typename uniform_int_distribution<IntType>::param_type& x, const typename uniform_int_distribution<IntType>::param_type& y)
	{
		return x.a == y.a && x.b == y.b;
	}

	template <class IntType>
	GPU_DEVICE bool operator!=(const typename uniform_int_distribution<IntType>::param_type& x, const typename uniform_int_distribution<IntType>::param_type& y)
	{
		return !(x == y);
	}

	template <class IntType>
	GPU_DEVICE uniform_int_distribution<IntType>::uniform_int_distribution(IntType a, IntType b) :
		m_p(param_type(a, b))
	{
	}

	template <class IntType>
	GPU_DEVICE uniform_int_distribution<IntType>::uniform_int_distribution(const param_type& p) :
		m_p(p)
	{
	}

	template <class IntType>
	GPU_DEVICE uniform_int_distribution<IntType>::result_type uniform_int_distribution<IntType>::a() const
	{
		return m_p.a();
	}

	template <class IntType>
	GPU_DEVICE uniform_int_distribution<IntType>::result_type uniform_int_distribution<IntType>::b() const
	{
		return m_p.b();
	}

	template <class IntType>
	GPU_DEVICE uniform_int_distribution<IntType>::result_type uniform_int_distribution<IntType>::max() const
	{
		return b();
	}

	template <class IntType>
	GPU_DEVICE uniform_int_distribution<IntType>::result_type uniform_int_distribution<IntType>::min() const
	{
		return a();
	}

	template <class IntType>
	template <class URNG>
	GPU_DEVICE uniform_int_distribution<IntType>::result_type uniform_int_distribution<IntType>::operator()(URNG& g)
	{
		return (*this)(g, m_p);
	}

	template <class IntType>
	template <class URNG>
	GPU_DEVICE uniform_int_distribution<IntType>::result_type uniform_int_distribution<IntType>::operator()(URNG& g, const param_type& p)
	{
		result_type v = g();
		return v % p.b() + p.a();
	}

	template <class IntType>
	GPU_DEVICE uniform_int_distribution<IntType>::param_type uniform_int_distribution<IntType>::param() const
	{
		return m_p;
	}

	template <class IntType>
	GPU_DEVICE void uniform_int_distribution<IntType>::param(const param_type& p)
	{
		m_p = p;
	}

	template <class IntType>
	GPU_DEVICE void uniform_int_distribution<IntType>::reset()
	{
	}

	template <class IntType>
	GPU_DEVICE bool operator==(const uniform_int_distribution<IntType>& x, const uniform_int_distribution<IntType>& y)
	{
		return x.p == y.p;
	}

	template <class IntType>
	GPU_DEVICE bool operator!=(const uniform_int_distribution<IntType>& x, const uniform_int_distribution<IntType>& y)
	{
		return !(x == y);
	}
}
