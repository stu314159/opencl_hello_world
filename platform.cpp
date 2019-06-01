#include <CL/cl.h>
#include <iostream>

int main(int argc, char* argv[]){
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_int status = clGetPlatformIDs(0,NULL,&numPlatforms);

  std::cout << "There are " << numPlatforms << " ocl platforms" << std::endl;

  // my laptop has 1 platform; investigate it
  cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
  status = clGetPlatformIDs(numPlatforms,platforms,NULL);
  platform = platforms[0];
  
  // devices
  cl_uint numDevices = 0;
  cl_device_id * devices;
  status = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&numDevices);

  std::cout << "There are " << numDevices << " ocl GPU devices " << std::endl;


  return 0;
}
