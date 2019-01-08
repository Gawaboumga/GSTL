#include <gstl/kernel/launch_kernel.cuh>

namespace gpu
{
	namespace kernel
	{
		template <class Function, class... Args>
		inline bool launch_kernel(Function f, cuda::launch_configuration_t launch_configuration, Args&&... args)
		{
			cuda::launch(
				f,
				launch_configuration,
				std::forward<Args>(args)...
			);

			auto status = cuda::outstanding_error::get();
			cuda::throw_if_error(status, "Failed to launch kernel");

			return true;
		}

		inline bool sync()
		{
			auto status = cuda::outstanding_error::get();
			cuda::throw_if_error(status, "Failed to launch kernel");

			cuda::device::current::get().synchronize();

			status = cuda::outstanding_error::get();
			cuda::throw_if_error(status, "Failed to launch kernel");

			return true;
		}
	}
}
