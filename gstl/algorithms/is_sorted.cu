#include <gstl/algorithms/is_sorted.cuh>

#include <gstl/algorithms/adjacent_find.cuh>
#include <gstl/functional/function_object.cuh>
#include <gstl/utility/iterator.cuh>

namespace gpu
{
	template <class ForwardIt>
	GPU_DEVICE bool is_sorted(block_t g, ForwardIt first, ForwardIt last)
	{
		return is_sorted_until(g, first, last) == last;
	}

	template <class BlockTile, class ForwardIt>
	GPU_DEVICE bool is_sorted(BlockTile g, ForwardIt first, ForwardIt last)
	{
		return is_sorted_until(g, first, last) == last;
	}

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted(ForwardIt first, ForwardIt last)
	{
		return is_sorted_until(first, last) == last;
	}

	template <class ForwardIt, class Compare>
	GPU_DEVICE bool is_sorted(block_t g, ForwardIt first, ForwardIt last, Compare comp)
	{
		return is_sorted_until(g, first, last, comp) == last;
	}

	template <class BlockTile, class ForwardIt, class Compare>
	GPU_DEVICE bool is_sorted(BlockTile g, ForwardIt first, ForwardIt last, Compare comp)
	{
		return is_sorted_until(g, first, last, comp) == last;
	}

	template <class ForwardIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR bool is_sorted(ForwardIt first, ForwardIt last, Compare comp)
	{
		return is_sorted_until(first, last, comp) == last;
	}

	template <class ForwardIt>
	GPU_DEVICE ForwardIt is_sorted_until(block_t g, ForwardIt first, ForwardIt last)
	{
		return is_sorted_until(g, first, last, less<>());
	}

	template <class BlockTile, class ForwardIt>
	GPU_DEVICE ForwardIt is_sorted_until(BlockTile g, ForwardIt first, ForwardIt last)
	{
		return is_sorted_until(g, first, last, less<>());
	}

	template <class ForwardIt>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt is_sorted_until(ForwardIt first, ForwardIt last)
	{
		return is_sorted_until(first, last, less<>());
	}

	template <class ForwardIt, class Compare>
	GPU_DEVICE ForwardIt is_sorted_until(block_t g, ForwardIt first, ForwardIt last, Compare comp)
	{
		ForwardIt it = adjacent_find(g, first, last, [&comp](const auto& lhs, const auto& rhs) {
			return comp(lhs, rhs);
		});
		return it == last ? last : next(it);
	}

	template <class BlockTile, class ForwardIt, class Compare>
	GPU_DEVICE ForwardIt is_sorted_until(BlockTile g, ForwardIt first, ForwardIt last, Compare comp)
	{
		ForwardIt it = adjacent_find(g, first, last, [&comp](const auto& lhs, const auto& rhs) {
			return comp(lhs, rhs);
		});
		return it == last ? last : next(it);
	}

	template <class ForwardIt, class Compare>
	GPU_DEVICE GPU_CONSTEXPR ForwardIt is_sorted_until(ForwardIt first, ForwardIt last, Compare comp)
	{
		ForwardIt it = adjacent_find(first, last, [&comp](const auto& lhs, const auto& rhs) {
			return comp(lhs, rhs);
		});
		return it == last ? last : next(it);
	}
}
