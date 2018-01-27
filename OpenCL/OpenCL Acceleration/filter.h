
#ifndef _FILTER_H_

#define _FILTER_H_

// OpenCL Data types
// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html

typedef struct 
{
  // OpenCL Setup
  cl_platform_id platform;
  cl_program program;
  cl_kernel kernel;
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;


  // Specific Filter 
  //   Image data
  cl_mem bufferImageIn;
  cl_mem bufferImageOut;
  //   Filter data
  //cl_mem bufferImageKernel;

  // Problem space concepts 
  int Height;
  int Width;
  int kernelRadius;
  int boost;

  // Parameters for running
  int filterMode;
  char *filterName;
  char *bitmapInputFile;
  char *bitmapOutputFile;
  char *openclKernelFile;

  // Controlling the image data
  unsigned char *imageIn;
  unsigned char *imageOut;
  char *imageKernel;
} filter;

#define clear() printf("\033[H\033[J")
#define KERNELSIZE(radius) (((radius)*2 + 1)* ((radius)*2 + 1))
struct timeval timerStart;

void StartTimer() {
	gettimeofday(&timerStart, NULL);
}

double GetTimer() {
	struct timeval timerStop, timerElapsed;
	gettimeofday(&timerStop, NULL);
	timersub(&timerStop, &timerStart, &timerElapsed);
	return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}


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


void cl_init(filter *self) 
{
  cl_int ret;

  // Get a platform - pick the first one found
 
  // Only looking for 1 platform
  ret = clGetPlatformIDs(1, &self->platform, NULL);
  checkResult(ret);

  // Only looking for 1 device 
  unsigned int noOfDevices;
  ret = clGetDeviceIDs(self->platform, CL_DEVICE_TYPE_CPU, 1, &self->device, &noOfDevices);
  checkResult(ret);

  // Create a context and command queue on that device
  self->context = clCreateContext(NULL, 1, &self->device, NULL, NULL, &ret);
  checkResult(ret);

  cl_device_type t;
  self->queue = clCreateCommandQueue(self->context, self->device, 0, &ret);
  checkResult(ret);
  ret = clGetDeviceInfo(self->device, CL_DEVICE_TYPE, sizeof(t), &t, NULL);
  checkResult(ret);
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


const char *filter_read_source(const char *kernelFilename)
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


void filter_setup(filter *self)
{
   int ret;
   char clOptions[100];

   const char *source= NULL;
   source= filter_read_source(self->openclKernelFile);

   if (source == 0) {
     fprintf(stderr, "ERROR - in %s:%d: kernel source string for filter is NULL\n", __FILE__, __LINE__);
     exit(-1);
   }

   // Perform runtime source compilation, and obtain kernel entry point.
   self->program = clCreateProgramWithSource(self->context, 1, &source, NULL, &ret);
   checkResult(ret);
   free((char*)source);

   sprintf(clOptions, "-DHEIGHT=%d -DWIDTH=%d -DKERNEL_RADIUS=%d", self->Height, self->Width, self->kernelRadius);
   //sprintf(clOptions, "-DHEIGHT=%d -DWIDTH=%d -DKERNEL_RADIUS=%d", self->Height, self->Width, self->kernelRadius);
   ret = clBuildProgram(self->program, 1, &self->device,NULL, NULL, NULL );
   //STATUSCHKMSG("build");
   if (ret != CL_SUCCESS)
   {
        size_t len;
        char buffer[3200000];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(self->program, self->device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        checkResult(ret);
   }


   self->kernel = clCreateKernel(self->program, self->filterName, &ret);
   checkResult(ret);

   // Create data buffers
   // Input
   self->bufferImageIn = clCreateBuffer(self->context, CL_MEM_READ_ONLY, self->Height * self->Width * sizeof(cl_uchar), NULL, &ret);
   checkResult(ret);

   // Output
   self->bufferImageOut = clCreateBuffer(self->context, CL_MEM_WRITE_ONLY, self->Height * self->Width * sizeof(cl_uchar), NULL, &ret);
   checkResult(ret);

   // Create Kernel size : 3x3 or 5x5
   //self->bufferImageKernel = clCreateBuffer(self->context, CL_MEM_READ_ONLY,  KERNELSIZE(self->kernelRadius) * sizeof(cl_char), NULL, &ret);
   //checkResult(ret);

   // Set the kernel arguments
   ret  = clSetKernelArg(self->kernel, 0, sizeof(self->bufferImageIn),  (void*) &self->bufferImageIn);
   ret |= clSetKernelArg(self->kernel, 1, sizeof(self->bufferImageOut), (void*) &self->bufferImageOut);
   //ret |= clSetKernelArg(self->kernel, 2, sizeof(self->bufferImageKernel),   (void*) &self->bufferImageKernel);
   ret |= clSetKernelArg(self->kernel, 2, sizeof(int), &self->Height);
   ret |= clSetKernelArg(self->kernel, 3, sizeof(int), &self->Width);
   ret |= clSetKernelArg(self->kernel, 4, sizeof(int), &self->kernelRadius);
   checkResult(ret);
}


void filter_run(filter *self)
{
    // Launch the kernel. Let OpenCL pick the local work size
    cl_int ret;
    size_t global_work_size[2] = {self->Height, self->Width};

    // copy input data
    ret = clEnqueueWriteBuffer(self->queue, self->bufferImageIn, CL_TRUE, 0, 
          sizeof(cl_uchar)*self->Height*self->Width, self->imageIn, 0, NULL, NULL);
    checkResult(ret);

   // ret = clEnqueueWriteBuffer(self->queue, self->bufferImageKernel, CL_TRUE, 0, 
   //       sizeof(cl_uchar)*KERNELSIZE(self->kernelRadius), self->imageKernel, 0, NULL, NULL);
    //checkResult(ret);

    // Timing code start
	StartTimer();
    // launch the kernel
    ret = clEnqueueNDRangeKernel(self->queue, self->kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Wait until the device is finished
    clFinish(self->queue);

    // Timing code stop
	double runtime = GetTimer();
	printf("Finished with OPENCL Kernel\n");
	printf("Computation Time: %f ms\n\n", runtime);

    // Read back the results from the device to verify the output
    ret = clEnqueueReadBuffer(self->queue, self->bufferImageOut, CL_TRUE, 0, 
          sizeof(cl_uchar) * self->Height * self->Width, self->imageOut, 0, NULL, NULL);
    checkResult(ret);

}

void filter_close(filter *self) {
  clReleaseMemObject(self->bufferImageIn);
  clReleaseMemObject(self->bufferImageOut);
  //clReleaseMemObject(self->bufferImageKernel);

  clReleaseProgram(self->program);
  clReleaseCommandQueue(self->queue);
  clReleaseContext(self->context);
}

void filter_go(filter *self)
{
   // Talk to the OpenCL runtime layer, make sure it is there
   cl_init(self);

   // filter_setup(self, "filter_kernel.cl", "sobel");
   filter_setup(self);

   //  Run the code : global variables/allocated global arrays
   filter_run(self);

   // Clean up the OpenCL stuff
   filter_close(self);
}

#endif // _FILTER_H_


