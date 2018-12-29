#pragma once

#ifndef GPU_UTILITY_PAIR_HPP
#define GPU_UTILITY_PAIR_HPP

#include <gstl/prerequisites.hpp>

#include <utility>

namespace gpu
{
	template <typename T1, typename T2>
	struct pair
	{
		using first_type = T1;
		using second_type = T2;

		pair() = default;

		GPU_DEVICE pair(const first_type& x, const second_type& y) :
			first{ x },
			second{ y }
		{
		}

		GPU_DEVICE pair(first_type&& x, second_type&& y) :
			first{ std::forward<first_type>(x) },
			second{ std::forward<second_type>(y) }
		{
		}

		template <typename U1, typename U2>
		GPU_DEVICE pair(const pair<U1, U2>& other) :
			first{ other.first },
			second{ other.second }
		{
		}

		template <typename U1, typename U2>
		GPU_DEVICE pair(pair<U1, U2>&& other) :
			first{ std::forward<U1>(other.first) },
			second{ std::forward<U2>(other.second) }
		{
		}

		pair(const pair& other) = default;
		pair(pair&& other) = default;

		template <typename U1, typename U2>
		GPU_DEVICE pair& operator=(const pair<U1, U2>& other)
		{
			first = other.first;
			second = other.second;
			return *this;
		}

		template <typename U1, typename U2>
		GPU_DEVICE pair& operator=(pair<U1, U2>&& other)
		{
			first = std::forward<U1>(other.first);
			second = std::forward<U2>(other.second);
			return *this;
		}

		pair& operator=(const pair& other) = default;
		pair& operator=(pair&& other) = default;

		first_type first;
		second_type second;
	};

	template <typename T1, typename T2>
	GPU_DEVICE pair<T1, T2> make_pair(const T1& t, const T2& u)
	{
		return pair<T1, T2>(t, u);
	}

	template <typename T1, typename T2>
	GPU_DEVICE bool operator==(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs)
	{
		return lhs.first == rhs.first && lhs.second == rhs.second;
	}

	template <typename T1, typename T2>
	GPU_DEVICE bool operator!=(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs)
	{
		return !operator==(lhs, rhs);
	}
}

#endif // GPU_UTILITY_PAIR_HPP
