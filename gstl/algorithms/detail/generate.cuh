#pragma once

#ifndef GPU_ALGORITHMS_DETAIL_GENERATE_HPP
#define GPU_ALGORITHMS_DETAIL_GENERATE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	namespace detail
	{
		template <class Thread, class RandomIt, class Generator>
		GPU_DEVICE void generate(Thread g, RandomIt first, RandomIt last, Generator gen);

		template <class Thread, class ForwardIt, class Size, class Generator>
		GPU_DEVICE void generate_n(Thread g, ForwardIt first, Size n, Generator gen);
	}
}

#include <gstl/algorithms/detail/generate.cu>

#endif // GPU_ALGORITHMS_DETAIL_GENERATE_HPP
