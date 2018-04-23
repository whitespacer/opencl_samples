#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include <CL/cl.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {
      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      // load opencl source
      std::ifstream cl_file("reduce.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
      try
      {
         program.build(devices, "-DBLOCK_SIZE=256");
      }
      catch (cl::Error const & e)
      {
         std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
         std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
         std::cout << log_str;
         return 0;
      }

      size_t const block_size = 256;
      size_t const test_array_size = 256 * 100000;
      size_t const output_size = test_array_size / block_size;
      std::vector<int> input(test_array_size);
      std::vector<int> output(output_size, 0);
      int block_sum = block_size * (block_size - 1) / 2;
      for (size_t i = 0; i < test_array_size; ++i)
      {
         input[i] = i % block_size;
      }

      // allocate device buffer to hold message
      cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(int) * test_array_size);
      cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(int) * output_size);

      auto kernels = { "gpu_reduce_gmem", 
                       "gpu_reduce_lmem",
                       "gpu_reduce_lmem_interleaved_addressing",
                       "gpu_reduce_lmem_no_banks_conflicts",
                       "gpu_reduce_lmem_pragma_unroll",
                       "gpu_reduce_lmem_unroll"
      };
      for (auto const& kernel : kernels)
      {
         // copy from cpu to gpu
         queue.enqueueWriteBuffer(dev_input, CL_FALSE, 0, sizeof(int) * test_array_size, &input[0]);

         // load named kernel from opencl source
         cl::Kernel kernel_gmem(program, kernel);
         // Make kernel can be used here
         kernel_gmem.setArg(0, dev_input);
         kernel_gmem.setArg(1, dev_output);
         if (kernel != std::string("gpu_reduce_gmem"))
            kernel_gmem.setArg(2, cl::Local(sizeof(int)* block_size));

         cl::Event event;
         queue.enqueueNDRangeKernel(kernel_gmem,
            cl::NullRange,
            cl::NDRange(test_array_size),
            cl::NDRange(block_size),
            nullptr,
            &event);

         event.wait();
         cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
         cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
         cl_ulong elapsed_time = end_time - start_time;


         queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(int) * output_size, &output[0]);
         for (size_t i = 0; i < output_size; ++i)
            if (output[i] != block_sum)
               throw cl::Error(-1, "invalid result");

         std::cout << std::setprecision(3) << "Total time: " << elapsed_time / 1000000.0 << " ms" << std::endl;
      }
   }
   catch (cl::Error const & e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}