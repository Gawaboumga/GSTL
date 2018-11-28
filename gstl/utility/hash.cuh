#pragma once

#ifndef GPU_UTILITY_HASH_HPP
#define GPU_UTILITY_HASH_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	template <class Key>
	struct hash;
}

#include <gstl/utility/hash.cu>

#endif // GPU_UTILITY_HASH_HPP
