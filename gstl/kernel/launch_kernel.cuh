#pragma once

#ifndef GPU_KERNEL_LAUNCH_KERNEL_HPP
#define GPU_KERNEL_LAUNCH_KERNEL_HPP

#include <gstl/prerequisites.hpp>

#include <cuda/api_wrappers.hpp>

namespace gpu
{
	namespace kernel
	{
		template <class Function, class... Args>
		inline bool launch_kernel(Function f, cuda::launch_configuration_t launch_configuration, Args&&... args);

		inline bool sync();
	}
}

#include <gstl/kernel/launch_kernel.cu>

#endif // GPU_KERNEL_LAUNCH_KERNEL_HPP
