#include <iostream>
#include <cstdlib>
#include <cmath>
#include <CL/opencl.h>

const char * kernelSource =                           "\n"\
"__kernel void vecAdd( __global double * c,            \n"\
"                      __global double * a,            \n"\
"                      __global double * b,            \n"\
"                      const int n)                    \n"\
"{                                                     \n"\
"         //Get our global thread ID                   \n"\
"         int id = get_global_id(0);                   \n"\
"                                                      \n"\
"         // make sure we do not go out of bounds      \n"\
"         if(id < n)                                   \n"\
"             c[id] = a[id]+b[id];                     \n"\
"}                                                     \n";

int main(int argc, char* argv[]){

  // get input argument
  const int N = atoi(argv[1]); //a good person would check for the argument first

  // construct host arrays
  double *a = new double[N];
  double *b = new double[N];
  double *c = new double[N];

  // initialize host arrays
  for(int i=0;i<N; i++){
    a[i] = i;
    b[i] = i;
  }

  // declare opencl device buffers
  cl_mem a_d;
  cl_mem b_d;
  cl_mem c_d;

  // declare some opencl objects
  cl_platform_id cpPlatform;
  cl_device_id device_id;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;

  size_t globalSize, localSize;
  cl_int err;
  
  // configure the opencl "thread grid:
  localSize=64;
  globalSize = ceil(N/(float)localSize)*localSize;

  // "bind" to platform
  err = clGetPlatformIDs(1,&cpPlatform,NULL);
  // get ID for the device
  err = clGetDeviceIDs(cpPlatform,CL_DEVICE_TYPE_GPU,1,&device_id,NULL);

  // create a context
  context = clCreateContext(0,1,&device_id,NULL,NULL,&err);

  // create a command queue
  queue = clCreateCommandQueue(context,device_id,0,&err);

  // create the compute program from the source buffer
  program = clCreateProgramWithSource(context,1,(const char**) &kernelSource,NULL,&err);

  // build the program executable
  clBuildProgram(program,0,NULL,NULL,NULL,NULL);

  // create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program,"vecAdd",&err);

  // create the input and output arrays in device memory
  size_t bytes = N*sizeof(double);
  a_d = clCreateBuffer(context,CL_MEM_READ_ONLY,bytes,NULL,NULL);
  b_d = clCreateBuffer(context,CL_MEM_READ_ONLY,bytes,NULL,NULL);
  c_d = clCreateBuffer(context,CL_MEM_WRITE_ONLY,bytes,NULL,NULL);

  // write our data set into the input array in device memory
  err = clEnqueueWriteBuffer(queue,a_d,CL_TRUE,0,bytes,a,0,NULL,NULL);
  err = clEnqueueWriteBuffer(queue,b_d,CL_TRUE,0,bytes,b,0,NULL,NULL);

  // set the arguments to our compute kernel
  err = clSetKernelArg(kernel,0,sizeof(cl_mem),&c_d);
  err = clSetKernelArg(kernel,1,sizeof(cl_mem),&a_d);
  err = clSetKernelArg(kernel,2,sizeof(cl_mem),&b_d);
  err = clSetKernelArg(kernel,3,sizeof(int),&N);

  // Execute the kernel over the entire range of the data set
  err = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&globalSize,&localSize,0,NULL,NULL);

  // wait for the command queue to get serviced before reading back the result
  clFinish(queue);

  // read the results from the device
  clEnqueueReadBuffer(queue,c_d,CL_TRUE,0,bytes,c,0,NULL,NULL);

  // check the results
  double sum = 0.;
  for(int i=0;i<N;i++){
    sum+=c[i];
  }

  std::cout << "Sum should equal " << N*(N-1) << std::endl;
  std::cout << "Sum equals: " << sum << std::endl;

  // deallocate opencl resources
  clReleaseMemObject(a_d);
  clReleaseMemObject(b_d);
  clReleaseMemObject(c_d);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  delete [] a;
  delete [] b;
  delete [] c;
  return 0;
}
