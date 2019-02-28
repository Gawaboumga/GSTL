#include <gstl/random/sample.cuh>

#include <gstl/algorithms/generate.cuh>
#include <gstl/random/linear_congruential_engine.cuh>
#include <gstl/random/random_device.cuh>
#include <gstl/random/uniform_int_distribution.cuh>
#include <gstl/utility/limits.cuh>

namespace gpu
{
	template <class OutputIt, class Size>
	GPU_DEVICE void randint(block_t g, OutputIt d_first, Size size)
	{
		using result_type = std::decay_t<decltype(*d_first)>;
		return randint(g, d_first, size, numeric_limits<result_type>::min(), numeric_limits<result_type>::max());
	}

	template <class OutputIt, class Size, unsigned int tile_sz>
	GPU_DEVICE void randint(block_tile_t<tile_sz> g, OutputIt d_first, Size size)
	{
		using result_type = std::decay_t<decltype(*d_first)>;
		return randint(g, d_first, size, numeric_limits<result_type>::min(), numeric_limits<result_type>::max());
	}

	template <class OutputIt, class Size, class Integral>
	GPU_DEVICE void randint(block_t g, OutputIt d_first, Size size, Integral low)
	{
		return randint(g, d_first, size, low, numeric_limits<Integral>::max());
	}

	template <class OutputIt, class Size, class Integral, unsigned int tile_sz>
	GPU_DEVICE void randint(block_tile_t<tile_sz> g, OutputIt d_first, Size size, Integral low)
	{
		return randint(g, d_first, size, low, numeric_limits<Integral>::max());
	}

	template <class OutputIt, class Size, class Integral>
	GPU_DEVICE void randint(block_t g, OutputIt d_first, Size size, Integral low, Integral high)
	{
		random_device rd;
		minstd_rand gen(rd());
		uniform_int_distribution<> dis(low, high);

		generate_n(g, d_first, size, [&dis, &gen]() {
			return dis(gen);
		});
	}

	template <class OutputIt, class Size, class Integral, unsigned int tile_sz>
	GPU_DEVICE void randint(block_tile_t<tile_sz> g, OutputIt d_first, Size size, Integral low, Integral high)
	{
		random_device rd;
		minstd_rand gen(rd());
		uniform_int_distribution<> dis(low, high);

		generate_n(g, d_first, size, [&dis, &gen]() {
			return dis(gen);
		});
	}
}
