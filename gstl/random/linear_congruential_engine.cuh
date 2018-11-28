#pragma once

#ifndef GPU_RANDOM_LINEAR_CONGRUENTIAL_ENGINE_HPP
#define GPU_RANDOM_LINEAR_CONGRUENTIAL_ENGINE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class UIntType, UIntType a, UIntType c, UIntType m>
	class linear_congruential_engine
	{
		public:
			using result_type = UIntType;

			static GPU_CONSTEXPR result_type multiplier = a;
			static GPU_CONSTEXPR result_type increment = c;
			static GPU_CONSTEXPR result_type modulus = m;
			static GPU_CONSTEXPR result_type default_seed = 1u;

			explicit GPU_DEVICE linear_congruential_engine(result_type s = default_seed);

			GPU_DEVICE void discard(unsigned long long z);

			GPU_DEVICE result_type operator()();

			GPU_DEVICE void seed(result_type = default_seed);

			static GPU_DEVICE GPU_CONSTEXPR result_type max();
			static GPU_DEVICE GPU_CONSTEXPR result_type min();

			GPU_DEVICE friend bool operator==(const linear_congruential_engine& x, const linear_congruential_engine& y);
			GPU_DEVICE friend bool operator!=(const linear_congruential_engine& x, const linear_congruential_engine& y);

		private:
			result_type m_x;

			static GPU_CONSTEXPR const result_type Mp = result_type(~0);
	};

	using minstd_rand0 = linear_congruential_engine<std::uint_fast32_t, 16807, 0, 2147483647>;
	using minstd_rand = linear_congruential_engine<std::uint_fast32_t, 48271, 0, 2147483647>;
}

#include <gstl/random/linear_congruential_engine.cu>

#endif // GPU_RANDOM_LINEAR_CONGRUENTIAL_ENGINE_HPP
