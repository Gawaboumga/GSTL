#include <gstl/utility/bits.cuh>

namespace gpu
{
	template <typename T>
	GPU_DEVICE T bits::extract(T data, offset_type begin, offset_type end)
	{
	#if defined(GPU_DEBUG_BITS)
		ENSURE(end > begin);
	#endif // GPU_DEBUG_BITS

		mask_type mask = (1 << end) - 1;
		unsigned_type lower_bits = to_unsigned(data) & mask;
		return lower_bits >> begin;
	}

	template <typename T>
	GPU_DEVICE T bits::from_unsigned(unsigned_type data)
	{
		return static_cast<T>(data);
	}

	template <typename T>
	GPU_DEVICE bits::unsigned_type bits::to_unsigned(T data)
	{
		return static_cast<unsigned_type>(data);
	}
}
