#pragma once

#ifndef GPU_ALGORITHMS_SORT_TAG_HPP
#define GPU_ALGORITHMS_SORT_TAG_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	struct arbitrary_tag {};
	struct greater_than_tag : public arbitrary_tag {};
	struct less_than_tag : public arbitrary_tag {};
	struct power_of_two_tag : public arbitrary_tag {};
	struct equal_tag : public arbitrary_tag {};
}

#endif // GPU_ALGORITHMS_SORT_TAG_HPP
