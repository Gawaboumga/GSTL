#pragma once

#ifndef GPU_RANDOM_UNIFORM_INT_DISTRIBUTION_HPP
#define GPU_RANDOM_UNIFORM_INT_DISTRIBUTION_HPP

#include <gstl/prerequisites.hpp>
#include <gstl/utility/limits.cuh>

namespace gpu
{
	template <class IntType = int>
	class uniform_int_distribution
	{
		public:
			using result_type = IntType;

			class param_type
			{
				public:
					using distribution_type = uniform_int_distribution;

					GPU_DEVICE explicit param_type(IntType a = 0, IntType b = numeric_limits<IntType>::max());

					GPU_DEVICE result_type a() const;

					GPU_DEVICE result_type b() const;

					GPU_DEVICE friend bool operator==(const param_type& x, const param_type& y);
					GPU_DEVICE friend bool operator!=(const param_type& x, const param_type& y);

				private:
					result_type m_a;
					result_type m_b;
			};

			GPU_DEVICE explicit uniform_int_distribution(IntType a = 0, IntType b = numeric_limits<IntType>::max());
			GPU_DEVICE explicit uniform_int_distribution(const param_type& parm);

			GPU_DEVICE result_type a() const;

			GPU_DEVICE result_type b() const;

			GPU_DEVICE result_type max() const;
			GPU_DEVICE result_type min() const;

			template <class URNG>
			GPU_DEVICE result_type operator()(URNG& g);
			template<class URNG>
			GPU_DEVICE result_type operator()(URNG& g, const param_type& parm);

			GPU_DEVICE param_type param() const;
			GPU_DEVICE void param(const param_type& parm);

			GPU_DEVICE void reset();

			GPU_DEVICE friend bool operator==(const uniform_int_distribution& x, const uniform_int_distribution& y);
			GPU_DEVICE friend bool operator!=(const uniform_int_distribution& x, const uniform_int_distribution& y);

		private:
			param_type m_p;
	};
}

#include <gstl/random/uniform_int_distribution.cu>

#endif // GPU_RANDOM_UNIFORM_INT_DISTRIBUTION_HPP
