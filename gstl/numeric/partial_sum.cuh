#ifndef GPU_NUMERIC_PARTIALSUM_HPP
#define GPU_NUMERIC_PARTIALSUM_HPP

namespace gpu
{
	template <class InputIt, class OutputIt>
	GPU_DEVICE GPU_CONSTEXPR OutputIt partial_sum(InputIt first, InputIt last, OutputIt d_first);

	template< class InputIt, class OutputIt, class BinaryOperation>
	GPU_DEVICE GPU_CONSTEXPR OutputIt partial_sum(InputIt first, InputIt last, OutputIt d_first, BinaryOperation op);
}

#include "partial_sum.cu"

#endif // GPU_NUMERIC_PARTIALSUM_HPP
