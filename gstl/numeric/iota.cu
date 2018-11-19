#include "iota.cuh"

namespace gpu
{
	template <class InputIt, typename T>
	GPU_DEVICE GPU_CONSTEXPR void iota(InputIt first, InputIt last, T value)
	{
		while (first != last)
		{
			*first++ = value;
			++value;
		}
	}
}
