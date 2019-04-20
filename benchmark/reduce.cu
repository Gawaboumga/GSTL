#include <cuda/api_wrappers.hpp>
#include <gstl/kernel/numeric/reduce.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>

#include <hayai.hpp>

constexpr unsigned int NUMBER_OF_RUNS = 1;
constexpr unsigned int NUMBER_OF_ITERATIONS = 1;
constexpr unsigned int NUMBER_OF_ELEMENTS = 1 << 28;
constexpr unsigned int BLOCKS_PER_GRID = 256;
constexpr unsigned int THREADS_PER_BLOCK = 1024;
using REDUCTION_TYPE = int;

struct prg_int
{
	int a, b;
	uint32_t seed;

	__host__ __device__ prg_int(int _a = 0, int _b = 255, uint32_t _seed = 1) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__ int operator()(const unsigned int n) const
	{
		thrust::minstd_rand rng(seed);
		thrust::uniform_int_distribution<int> dist(a, b);
		rng.discard(n);

		return dist(rng);
	}
};

struct prg_float
{
	float a, b;
	uint32_t seed;

	__host__ __device__ prg_float(float _a = 0.f, float _b = 1.f, uint32_t _seed = 1) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__ float operator()(const unsigned int n) const
	{
		thrust::minstd_rand rng(seed);
		thrust::uniform_real_distribution<float> dist(a, b);
		rng.discard(n);

		return dist(rng);
	}
};

template <unsigned int input_count, typename T>
class GSTLFixture : public ::hayai::Fixture
{
	public:
		virtual void SetUp() override
		{
			input = thrust::device_vector<T>(input_count);

			thrust::counting_iterator<unsigned int> index_sequence_begin(0);
			thrust::transform(index_sequence_begin, index_sequence_begin + input_count, input.begin(), prg_int(0, 255));
		}

		void benchmark_reduce()
		{
			cuda::launch_configuration_t configuration{ BLOCKS_PER_GRID, THREADS_PER_BLOCK };
			auto in = thrust::raw_pointer_cast(input.data());
			gpu::kernel::reduce(configuration, in, in + input_count);
			cuda::device::current::get().synchronize();
		}

		virtual void TearDown()
		{

		}

	private:
		thrust::device_vector<T> input;
};

template <unsigned int input_count, typename T>
class ThrustFixture : public ::hayai::Fixture
{
	public:
		virtual void SetUp()
		{
			input = thrust::device_vector<T>(input_count);

			thrust::counting_iterator<unsigned int> index_sequence_begin(0);
			thrust::transform(index_sequence_begin, index_sequence_begin + input_count, input.begin(), prg_int(0, 255));
		}

		void benchmark_reduce()
		{
			thrust::reduce(input.begin(), input.end());
		}

	private:
		thrust::device_vector<T> input;
};

using GSTL1000 = GSTLFixture<NUMBER_OF_ELEMENTS, REDUCTION_TYPE>;

BENCHMARK_F(GSTL1000, GSTL, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	benchmark_reduce();
}

using Thrust1000 = ThrustFixture<NUMBER_OF_ELEMENTS, REDUCTION_TYPE>;

BENCHMARK_F(Thrust1000, Thrust, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	benchmark_reduce();
}

