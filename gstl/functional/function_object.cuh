#pragma once

#ifndef GPU_FUNCTIONAL_FUNCTION_OBJECT_HPP
#define GPU_FUNCTIONAL_FUNCTION_OBJECT_HPP

#include <gstl/prerequisites.hpp>

namespace gpu
{
	struct identity;

	template <class T = void>
	struct plus;

	template <class T = void>
	struct minus;

	template <class T = void>
	struct multiplies;

	template <class T = void>
	struct divides;

	template <class T = void>
	struct modulus;

	template <class T = void>
	struct negate;

	template <class T = void>
	struct equal_to;

	template <class T = void>
	struct not_equal_to;

	template <class T = void>
	struct greater;

	template <class T = void>
	struct less;

	template <class T = void>
	struct greater_equal;

	template <class T = void>
	struct less_equal;

	template <class T = void>
	struct logical_and;

	template <class T = void>
	struct logical_or;

	template <class T = void>
	struct logical_not;

	template <class T = void>
	struct bit_and;

	template <class T = void>
	struct bit_or;

	template <class T = void>
	struct bit_xor;

	template <class T = void>
	struct bit_not;
}

#include <gstl/functional/function_object.cu>

#endif // GPU_FUNCTIONAL_FUNCTION_OBJECT_HPP
