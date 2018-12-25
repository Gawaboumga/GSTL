#include <gstl/memory/algorithms.cuh>

#include <gstl/memory/misc.cuh>

namespace gpu
{
	namespace detail
	{
		template <class InputIt, class CreationFunction>
		GPU_DEVICE InputIt iterate_over(InputIt first, InputIt last, CreationFunction f)
		{
			for (; first != last; ++first)
				f(first);

			return first;
		}

		template <class Thread, class InputIt, class CreationFunction>
		GPU_DEVICE InputIt iterate_over(Thread g, InputIt first, InputIt last, CreationFunction f)
		{
			auto copy_first = first;

			auto len = distance(first, last);
			offset_t offset = 0;
			while (offset < len)
			{
				f(first + g.thread_rank());
				first += g.size();
				offset += g.size();
			}

			return first + len;
		}

		template <class InputIt, class Size, class CreationFunction>
		GPU_DEVICE InputIt iterate_over_n(InputIt first, Size n, CreationFunction f)
		{
			for (; n > 0; (void) ++first, --n)
				f(first);

			return first;
		}

		template <class Thread, class InputIt, class Size, class CreationFunction>
		GPU_DEVICE InputIt iterate_over_n(Thread g, InputIt first, Size n, CreationFunction f)
		{
			auto copy_first = first;

			offset_t offset = 0;
			while (offset < n)
			{
				f(first + g.thread_rank());
				first += g.size();
				offset += g.size();
			}

			return copy_first + n;
		}

		template <class InputIt, class ForwardIt, class CreationFunction>
		GPU_DEVICE ForwardIt iterate_over_pair(InputIt first, InputIt last, ForwardIt d_first, CreationFunction f)
		{
			for (; first != last; ++d_first, (void) ++first)
				f(first, d_first);

			return d_first;
		}

		template <class Thread, class InputIt, class ForwardIt, class CreationFunction>
		GPU_DEVICE ForwardIt iterate_over_pair(Thread g, InputIt first, InputIt last, ForwardIt d_first, CreationFunction f)
		{
			auto copy_d_first = d_first;

			auto len = distance(first, last);
			offset_t offset = 0;
			while (offset < len)
			{
				f(first + g.thread_rank(), d_first + g.thread_rank());
				first += g.size();
				d_first += g.size();
				offset += g.size();
			}

			return copy_d_first + len;
		}

		template <class InputIt, class Size, class ForwardIt, class CreationFunction>
		GPU_DEVICE ForwardIt iterate_over_pair_n(InputIt first, Size n, ForwardIt d_first, CreationFunction f)
		{
			for (; n > 0; ++d_first, (void) ++first, --n)
				f(first, d_first);

			return d_first;
		}

		template <class Thread, class InputIt, class Size, class ForwardIt, class CreationFunction>
		GPU_DEVICE ForwardIt iterate_over_pair_n(Thread g, InputIt first, Size count, ForwardIt d_first, CreationFunction f)
		{
			auto copy_d_first = d_first;

			offset_t offset = 0;
			while (offset < count)
			{
				f(first + g.thread_rank(), d_first + g.thread_rank());
				first += g.size();
				d_first += g.size();
				offset += g.size();
			}

			return copy_d_first + count;
		}
	}

	template <class InputIt, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_copy(InputIt first, InputIt last, ForwardIt d_first)
	{
		return detail::iterate_over_pair(first, last, d_first, [](auto it, auto d_it) {
			::new (static_cast<void*>(addressof(*d_it))) typename std::iterator_traits<ForwardIt>::value_type(*it);
		});
	}

	template <class Thread, class InputIt, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_copy(Thread g, InputIt first, InputIt last, ForwardIt d_first)
	{
		return detail::iterate_over_pair(g, first, last, d_first, [](auto it, auto d_it) {
			::new (static_cast<void*>(addressof(*d_it))) typename std::iterator_traits<ForwardIt>::value_type(*it);
		});
	}

	template <class InputIt, class Size, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_copy_n(InputIt first, Size count, ForwardIt d_first)
	{
		return detail::iterate_over_pair_n(first, count, d_first, [](auto it, auto d_it) {
			::new (static_cast<void*>(addressof(*d_it))) typename std::iterator_traits<ForwardIt>::value_type(*it);
		});
	}

	template <class Thread, class InputIt, class Size, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_copy_n(Thread g, InputIt first, Size count, ForwardIt d_first)
	{
		return detail::iterate_over_pair_n(g, first, count, d_first, [](auto it, auto d_it) {
			::new (static_cast<void*>(addressof(*d_it))) typename std::iterator_traits<ForwardIt>::value_type(*it);
		});
	}

	template <class ForwardIt, typename T>
	GPU_DEVICE void uninitialized_fill(ForwardIt first, ForwardIt last, const T& value)
	{
		detail::iterate_over(first, last, [&value](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type(value);
		});
	}

	template <class Thread, class ForwardIt, typename T>
	GPU_DEVICE void uninitialized_fill(Thread g, ForwardIt first, ForwardIt last, const T& value)
	{
		detail::iterate_over(g, first, last, [&value](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type(value);
		});
	}

	template <class ForwardIt, class Size, typename T>
	GPU_DEVICE ForwardIt uninitialized_fill_n(ForwardIt first, Size count, const T& value)
	{
		return detail::iterate_over_n(first, count, [&value](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type(value);
		});
	}

	template <class Thread, class ForwardIt, class Size, typename T>
	GPU_DEVICE ForwardIt uninitialized_fill_n(Thread g, ForwardIt first, Size count, const T& value)
	{
		return detail::iterate_over_n(g, first, count, [&value](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type(value);
		});
	}

	template <class InputIt, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_move(InputIt first, InputIt last, ForwardIt d_first)
	{
		return detail::iterate_over_pair(first, last, d_first, [](auto it, auto d_it) {
			::new (static_cast<void*>(addressof(*d_it))) typename std::iterator_traits<ForwardIt>::value_type(std::move(*it));
		});
	}

	template <class Thread, class InputIt, class ForwardIt>
	GPU_DEVICE ForwardIt uninitialized_move(Thread g, InputIt first, InputIt last, ForwardIt d_first)
	{
		return detail::iterate_over_pair(g, first, last, d_first, [](auto it, auto d_it) {
			::new (static_cast<void*>(addressof(*d_it))) typename std::iterator_traits<ForwardIt>::value_type(std::move(*it));
		});
	}

	template <class InputIt, class Size, class ForwardIt>
	GPU_DEVICE std::pair<InputIt, ForwardIt> uninitialized_move_n(InputIt first, Size count, ForwardIt d_first)
	{
		return detail::iterate_over_pair_n(first, count, d_first, [](auto it, auto d_it) {
			::new (static_cast<void*>(addressof(*d_it))) typename std::iterator_traits<ForwardIt>::value_type(std::move(*it));
		});
	}

	template <class Thread, class InputIt, class Size, class ForwardIt>
	GPU_DEVICE std::pair<InputIt, ForwardIt> uninitialized_move_n(Thread g, InputIt first, Size count, ForwardIt d_first)
	{
		return detail::iterate_over_pair_n(g, first, count, d_first, [](auto it, auto d_it) {
			::new (static_cast<void*>(addressof(*d_it))) typename std::iterator_traits<ForwardIt>::value_type(std::move(*it));
		});
	}

	template <class ForwardIt>
	GPU_DEVICE void uninitialized_default_construct(ForwardIt first, ForwardIt last)
	{
		detail::iterate_over(first, last, [](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type;
		});
	}

	template <class Thread, class ForwardIt>
	GPU_DEVICE void uninitialized_default_construct(Thread g, ForwardIt first, ForwardIt last)
	{
		detail::iterate_over(g, first, last, [](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type;
		});
	}

	template <class ForwardIt, class Size>
	GPU_DEVICE ForwardIt uninitialized_default_construct_n(ForwardIt first, Size n)
	{
		return detail::iterate_over_n(first, n, [](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type;
		});
	}

	template <class Thread, class ForwardIt, class Size>
	GPU_DEVICE ForwardIt uninitialized_default_construct_n(Thread g, ForwardIt first, Size n)
	{
		return detail::iterate_over_n(g, first, n, [](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type;
		});
	}

	template <class ForwardIt>
	GPU_DEVICE void uninitialized_value_construct(ForwardIt first, ForwardIt last)
	{
		detail::iterate_over(first, last, [](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type();
		});
	}

	template <class Thread, class ForwardIt>
	GPU_DEVICE void uninitialized_value_construct(Thread g, ForwardIt first, ForwardIt last)
	{
		detail::iterate_over(g, first, last, [](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type();
		});
	}

	template <class ForwardIt, class Size>
	GPU_DEVICE ForwardIt uninitialized_value_construct_n(ForwardIt first, Size n)
	{
		return detail::iterate_over_n(first, n, [](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type();
		});
	}

	template <class Thread, class ForwardIt, class Size>
	GPU_DEVICE ForwardIt uninitialized_value_construct_n(Thread g, ForwardIt first, Size n)
	{
		return detail::iterate_over_n(g, first, n, [](auto it) {
			::new (static_cast<void*>(addressof(*it))) typename std::iterator_traits<ForwardIt>::value_type();
		});
	}

	template <typename T>
	GPU_DEVICE void destroy_at(T* p)
	{
		p->~T();
	}

	template <class ForwardIt>
	GPU_DEVICE void destroy(ForwardIt first, ForwardIt last)
	{
		detail::iterate_over(first, last, [](auto it) {
			destroy_at(addressof(*it));
		});
	}

	template <class Thread, class ForwardIt>
	GPU_DEVICE void destroy(Thread g, ForwardIt first, ForwardIt last)
	{
		detail::iterate_over(g, first, last, [](auto it) {
			destroy_at(addressof(*it));
		});
	}

	template <class ForwardIt, class Size>
	GPU_DEVICE ForwardIt destroy_n(ForwardIt first, Size n)
	{
		return detail::iterate_over_n(first, n, [](auto it) {
			destroy_at(addressof(*it));
		});
	}

	template <class Thread, class ForwardIt, class Size>
	GPU_DEVICE ForwardIt destroy_n(Thread g, ForwardIt first, Size n)
	{
		return detail::iterate_over_n(g, first, n, [](auto it) {
			destroy_at(addressof(*it));
		});
	}
}
