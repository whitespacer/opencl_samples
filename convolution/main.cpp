#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#endif
#include "cl2.hpp"

#include <vector>
#include <fstream>
#include <iostream>
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
      std::ifstream cl_file("convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, cl_string);

      // create program
      cl::Program program(context, source);

      // compile opencl source
      program.build(devices);

      // create a message to send to kernel
      size_t const block_size = 128;
      size_t const test_array_size = 512;
      size_t const mask_size = 9;

      std::vector<int> input(test_array_size);
      std::vector<int> output(test_array_size, 1);
      for (size_t i = 0; i < test_array_size; ++i)
      {
         input[i] = 1;
      }
      int mask[mask_size] = { 1, 1, 1, 1, -1, 1, 1, 1, 1 };

      // allocate device buffer to hold message
      cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(int) * test_array_size);
      cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(int) * test_array_size);
      cl::Buffer dev_mask  (context, CL_MEM_READ_ONLY, sizeof(int) * mask_size);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(int) * test_array_size, &input[0]);
      queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, sizeof(int) * mask_size, &mask[0]);

      // load named kernel from opencl source
      cl::Kernel kernel_gmem(program, "gpu_convolution_gmem");
      cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int> convolution_gmem(kernel_gmem);
      cl::EnqueueArgs eargs(queue, cl::NDRange(test_array_size), cl::NDRange(block_size));
      cl::Event event = convolution_gmem(eargs, dev_input, dev_mask, dev_output, mask_size, test_array_size);

      // Uncomment to run kernel with local memory
      //cl::Kernel kernel_lmem(program, "gpu_convolution_lmem");
      //cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, cl::LocalSpaceArg> convolution_lmem(kernel_lmem);
      //cl::Event event = convolution_lmem(eargs, dev_input, dev_mask, dev_output, mask_size, test_array_size,
      //                                   cl::Local((block_size + mask_size) * sizeof(int)));

      event.wait();
      cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end_time   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      cl_ulong elapsed_time = end_time - start_time;

      queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(int) * test_array_size, &output[0]);
      for (size_t i = 0; i < test_array_size; ++i)
         std::cout << output[i] << std::endl;
      std::cout << std::endl;

      std::cout << std::setprecision(2) << "Total time: " << elapsed_time / 1000000.0 << " ms" << std::endl;

   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
