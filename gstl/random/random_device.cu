#include <gstl/random/random_device.cuh>

#include <gstl/utility/hash.cuh>
#include <gstl/utility/limits.cuh>

namespace gpu
{
	inline GPU_DEVICE double random_device::entropy() const noexcept
	{
		return 0.0;
	}

	inline GPU_DEVICE GPU_CONSTEXPR random_device::result_type random_device::max()
	{
		return numeric_limits<result_type>::max();
	}

	inline GPU_DEVICE GPU_CONSTEXPR random_device::result_type random_device::min()
	{
		return numeric_limits<result_type>::min();
	}

	inline GPU_DEVICE random_device::result_type random_device::operator()()
	{
		return hash<offset_t>()(threadIdx.x + blockIdx.x * blockDim.x);
	}
}
