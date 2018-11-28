#pragma once

#ifndef GPU_RANDOM_RANDOM_DEVICE_HPP
#define GPU_RANDOM_RANDOM_DEVICE_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	class random_device
	{
		public:
			using result_type = unsigned int;

			GPU_DEVICE random_device() = default;
			GPU_DEVICE random_device(const random_device&) = delete;

			GPU_DEVICE double entropy() const noexcept;

			static GPU_DEVICE GPU_CONSTEXPR result_type max();
			static GPU_DEVICE GPU_CONSTEXPR result_type min();

			GPU_DEVICE result_type operator()();
			GPU_DEVICE random_device& operator=(const random_device&) = delete;
	};
}

#include <gstl/random/random_device.cu>

#endif // GPU_RANDOM_RANDOM_DEVICE_HPP
