#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include <CL/cl.h>
#include "cl2.hpp"

#include <vector>
#include <fstream>
#include <iostream>

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
      std::ifstream cl_file("scan.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, cl_string);

      // create program
      cl::Program program(context, source);

      // compile opencl source
      try
      {
         program.build(devices);
      }
      catch (cl::Error const & e)
      {         
         std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
         std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
         std::cout << log_str;
         return 0;
      }

      size_t const block_size = 256;
      size_t const test_array_size = 512;
      size_t const output_size = test_array_size;
      std::vector<int> input(test_array_size);
      std::vector<int> output(output_size, 0);
      for (size_t i = 0; i < test_array_size; ++i)
      {
         input[i] = i % 10;
      }

      // allocate device buffer to hold message
      cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(int) * test_array_size);
      cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(int) * output_size);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(int) * test_array_size, &input[0]);

      // load named kernel from opencl source
      cl::Kernel kernel_hs(program, "scan_hillis_steele");
      kernel_hs.setArg(0, dev_input);
      kernel_hs.setArg(1, dev_output);
      kernel_hs.setArg(2, cl::Local(sizeof(int)* block_size));
      kernel_hs.setArg(3, cl::Local(sizeof(int)* block_size));
      queue.enqueueNDRangeKernel(kernel_hs, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size));

      //cl::Kernel kernel_bs(program, "scan_blelloch");
      //kernel_bs.setArg(0, dev_input);
      //kernel_bs.setArg(1, dev_output);
      //kernel_bs.setArg(2, cl::Local(sizeof(int)* block_size));
      //queue.enqueueNDRangeKernel(kernel_bs, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size));

      queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(int) * output_size, &output[0]);

      for (size_t i = 0; i < output_size; ++i)
      {
         std::cout << output[i] << std::endl;
      }
   }
   catch (cl::Error const & e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
