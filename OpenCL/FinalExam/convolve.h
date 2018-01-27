
#ifndef _CONVOLVE_H_

#define _CONVOLVE_H_

// OpenCL Data types
// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html
//
//
//


// Timing support
int TimerStartIndex = 0;
int TimerStopIndex = 0;
unsigned long long ExecTimerStart[5], ExecTimerStop[5];

static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

typedef struct 
{
  // OpenCL Setup
  cl_platform_id platform;
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
} clsystem;

typedef struct 
{
  char *openclKernelFile;
  cl_program program;
  cl_kernel kernel;

  // Specific Filter 
  //   Image data
  cl_mem bufferImageIn;
  cl_mem bufferImageOut;
  //   Filter data
  cl_mem bufferImageFilter;

  // Problem space concepts 
  int Height;
  int Width;
  int filterSize;

  // Control for local memory allocation
  int localsize;

  // Controlling the image data
  float *imageIn;
  float *imageOut;
  float *filter;
} convolve;


#define RGBA_SIZE 4
#define PI                              3.14159265


// Check Error Code.
//
// @errorCode Error code to be checked.
// @file File from which this function is called.
// @line Line in the file, from which this is called.
//
void checkResult_(int errorCode, const char* file, int line) {
  if (errorCode != CL_SUCCESS) { // or CL_COMPLETE - also 0
    fprintf(stderr, "ERROR - in %s:%d: %d\n", file, line, errorCode);
    exit(-1);
  }
}


void checkResult_  (int, const char*, int);
void checkEvent_   (cl_event, const char*, int);
#define checkResult(v) checkResult_(v, __FILE__, __LINE__)
#define checkEvent(v) checkEvent_(v, __FILE__, __LINE__)


void cl_init(clsystem *sys) 
{
  cl_int ret;

  // Get a platform - pick the first one found
 
  // Only looking for 1 platform
  ret = clGetPlatformIDs(1, &sys->platform, NULL);
  checkResult(ret);

  // Only looking for 1 device 
  unsigned int noOfDevices;
  ret = clGetDeviceIDs(sys->platform, CL_DEVICE_TYPE_CPU, 1, &sys->device, &noOfDevices);
  checkResult(ret);

  // Create a context and command queue on that device
  sys->context = clCreateContext(NULL, 1, &sys->device, NULL, NULL, &ret);
  checkResult(ret);

  cl_device_type t;
  sys->queue = clCreateCommandQueue(sys->context, sys->device, 0, &ret);
  checkResult(ret);
  ret = clGetDeviceInfo(sys->device, CL_DEVICE_TYPE, sizeof(t), &t, NULL);
  checkResult(ret);
}

void cl_close(clsystem *sys)
{
  clReleaseCommandQueue(sys->queue);
  clReleaseContext(sys->context);
}


FILE *open_file(const char *filename, struct stat *info) 
{
  FILE *in = NULL;

  if (info) {
    if (stat(filename, info)) {
      fprintf(stderr,"ERROR: Could not stat : %s\n", filename);
      exit(-1);
    }
  }

  if ((in=fopen(filename, "rb"))==NULL) {
     fprintf(stderr, "ERROR: Could not open file: '%s'\n", filename);
     exit(EXIT_FAILURE);
  }

  return in;
}


const char *convolve_read_source(const char *kernelFilename)
{
  struct stat buf;
  FILE* in = open_file(kernelFilename, &buf);

  size_t size = buf.st_size;
  char *src = (char*)malloc(size+1);

  // Read the file content
  int len=0;
  if ((len = fread((void *)src, 1, size, in)) != (int)size) {
    fprintf(stderr, "ERROR: Read was not completed : %d / %lu bytes\n", len, size);
    exit(EXIT_FAILURE);
  }
 
  // end-of-string
  src[len]='\0';
  fclose(in);
  return src;
}


void convolve_setup(convolve *self, clsystem *sys)
{
   int ret;
   char clOptions[512];

   const char *source= NULL;
   source= convolve_read_source(self->openclKernelFile);

   if (source == 0) {
     fprintf(stderr, "ERROR - in %s:%d: kernel source string for convolve is NULL\n", __FILE__, __LINE__);
     exit(-1);
   }

   // Perform runtime source compilation, and obtain kernel entry point.
   self->program = clCreateProgramWithSource(sys->context, 1, &source, NULL, &ret);
   checkResult(ret);
   free((char*)source);

   sprintf(clOptions, "-DIMAGE_W=%d -DIMAGE_H=%d -DFILTER_SIZE=%d -DHALF_FILTER_SIZE=%d -DTWICE_HALF_FILTER_SIZE=%d -DHALF_FILTER_SIZE_IMAGE_W=%d",
                self->Width,
                self->Height,
                self->filterSize,
                self->filterSize/2,
                (self->filterSize/2) * 2,
                (self->filterSize/2) * self->Width
        );

   ret = clBuildProgram(self->program, 1, &sys->device, clOptions, NULL, NULL );

   if (ret != CL_SUCCESS)
   {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(self->program, sys->device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        checkResult(ret);
   }


   self->kernel = clCreateKernel(self->program, "convolve", &ret);
   checkResult(ret);

   // Create data buffers
   // Input
   self->bufferImageIn = clCreateBuffer(sys->context, CL_MEM_READ_ONLY, self->Height * self->Width * sizeof(cl_float) * RGBA_SIZE, NULL, &ret);
   checkResult(ret);

   // Output
   self->bufferImageOut = clCreateBuffer(sys->context, CL_MEM_WRITE_ONLY, self->Height * self->Width * sizeof(cl_float) * RGBA_SIZE, NULL, &ret);
   checkResult(ret);

   // Create Kernel size : 3x3 or 5x5, etc
   self->bufferImageFilter = clCreateBuffer(sys->context, CL_MEM_READ_ONLY,  self->filterSize* self->filterSize * sizeof(cl_float) * RGBA_SIZE, NULL, &ret);
   checkResult(ret);

   // Set the kernel arguments : only 4 arguments
   ret  = clSetKernelArg(self->kernel, 0, sizeof(self->bufferImageIn),  (void*) &self->bufferImageIn);
   ret |= clSetKernelArg(self->kernel, 1, sizeof(self->bufferImageOut), (void*) &self->bufferImageOut);
   ret |= clSetKernelArg(self->kernel, 2, sizeof(self->bufferImageFilter),   (void*) &self->bufferImageFilter);

   int localMemSize = 0;
   localMemSize = (self->localsize + 2 * (self->filterSize / 2)) *
                  (self->localsize + 2 * (self->filterSize / 2));

   localMemSize =  localMemSize * RGBA_SIZE * sizeof(cl_float);

   ret |= clSetKernelArg(self->kernel, 3, localMemSize, NULL);
   checkResult(ret);
}


void convolve_run(convolve *self, clsystem *sys)
{
    // Launch the kernel. Let OpenCL pick the local work size
    cl_int ret;
    size_t global_work_size[2] = {self->Width, self->Height};
    size_t local_work_size[2] = {self->localsize, self->localsize};

    // copy input data
    ret = clEnqueueWriteBuffer(sys->queue, self->bufferImageIn, CL_TRUE, 0, 
          sizeof(cl_float)*self->Height*self->Width * RGBA_SIZE, self->imageIn, 0, NULL, NULL);
    checkResult(ret);

    ret = clEnqueueWriteBuffer(sys->queue, self->bufferImageFilter, CL_TRUE, 0, 
          sizeof(cl_float)*self->filterSize*self->filterSize* RGBA_SIZE, self->filter, 0, NULL, NULL);
    checkResult(ret);

    // Timing code start

    // launch the kernel
    ret = clEnqueueNDRangeKernel(sys->queue, self->kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    // Wait until the device is finished
    clFinish(sys->queue);

    // Timing code stop

    // Read back the results from the device to verify the output
    ret = clEnqueueReadBuffer(sys->queue, self->bufferImageOut, CL_TRUE, 0, 
          sizeof(cl_float) * self->Height * self->Width * RGBA_SIZE, self->imageOut, 0, NULL, NULL);
    checkResult(ret);

}


void convolve_close(convolve *self) {
  clReleaseProgram(self->program);
  clReleaseMemObject(self->bufferImageIn);
  clReleaseMemObject(self->bufferImageOut);
  clReleaseMemObject(self->bufferImageFilter);
}

void convolve_go(convolve *self,  clsystem *sys)
{
   // convolve_setup(self, "convolve_kernel.cl", "sobel");
   convolve_setup(self, sys);

   //  Run the code : global variables/allocated global arrays
   convolve_run(self, sys);

   // Clean up the OpenCL stuff
   convolve_close(self);
}

#endif // _CONVOLVE_H_


