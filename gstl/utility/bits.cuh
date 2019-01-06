#pragma once

#ifndef GPU_UTILITY_BITS_HPP
#define GPU_UTILITY_BITS_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	struct bits
	{
		using mask_type = offset_t;
		using offset_type = offset_t;
		using unsigned_type = UI32;

		template <typename T>
		GPU_DEVICE static T extract(T data, offset_type begin, offset_type end);

		private:
			template <typename T>
			GPU_DEVICE static T from_unsigned(unsigned_type data);

			template <typename T>
			GPU_DEVICE static unsigned_type to_unsigned(T data);
	};
}

#include <gstl/utility/bits.cu>

#endif // GPU_UTILITY_BITS_HPP
