// 
// Convolve
//


// Includes: system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <string.h>

// OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#include "convolve.h"

// These macros control the entire execution of the image

#ifndef IMAGE_W
  #define IMAGE_W 512
#endif

#ifndef IMAGE_H
  #define IMAGE_H 512
#endif

#ifndef FILTER_W
  #define FILTER_W 7
#endif


#define HALF_FILTER_SIZE FILTER_W/2


static const float gaussian3[3*3] = {
        1.0 / 16.0,     2.0 / 16.0,     1.0 / 16.0,
        2.0 / 16.0,     4.0 / 16.0,     2.0 / 16.0,
        1.0 / 16.0,     2.0 / 16.0,     1.0 / 16.0
};

static const float gaussian5[5*5] = {
        1.0 / 273.0,    4.0 / 273.0,    7.0 / 273.0,    4.0 / 273.0,    1.0 / 273.0,
        4.0 / 273.0,    16.0 / 273.0,   26.0 / 273.0,   16.0 / 273.0,   4.0 / 273.0,
        7.0 / 273.0,    26.0 / 273.0,   41.0 / 273.0,   26.0 / 273.0,   7.0 / 273.0,
        4.0 / 273.0,    16.0 / 273.0,   26.0 / 273.0,   16.0 / 273.0,   4.0 / 273.0,
        1.0 / 273.0,    4.0 / 273.0,    7.0 / 273.0,    4.0 / 273.0,    1.0 / 273.0,
};

float *AllocateImage(int Width, int Height, int size) 
{
  return ((float *)malloc(sizeof(float) * Width * Height * size));
}


float *AllocateFilter(int filterSize, int size)
{
        const float DELTA = 1.84089642f * ((float) filterSize / 7.0);
        const float TWO_DELTA_SQ = 2.0f * DELTA * DELTA;
        const float k = 1.0f / (PI * TWO_DELTA_SQ);

        float *filter = (float *)malloc(filterSize * filterSize * sizeof(float)*size);
        float * fP = filter;
        int w = filterSize / 2;

        const float * precomputed = NULL;

        if (filterSize == 3)
        {
                precomputed = gaussian3;
        }
        else if (filterSize == 5)
        {
                precomputed = gaussian5;
        }


        for (int r = -w; r <= w; r++)
        {
                for (int c = -w; c <= w; c++, fP += 4)
                {
                        if (precomputed)
                        {
                                fP[0] = *precomputed;
                                fP[1] = *precomputed;
                                fP[2] = *precomputed;
                                fP[3] = r == c && c == 0 ? 1.0f : 0.0f;

                                precomputed++;
                        }
                        else
                        {
                                const float v = k * exp( -(r*r + c*c) / TWO_DELTA_SQ );
                                fP[0] = v;
                                fP[1] = v;
                                fP[2] = v;
                                fP[3] = r == c && c == 0 ? 1.0f : 0.0f;
                        }
                }

        }

  return filter;
}







// Functions
void Cleanup(void);

void CPU_Convolve(float *input, float *output, float *filter)
{
  // Go over image rows (height)
  for (int get_global_id1 = 0; get_global_id1 < IMAGE_H; get_global_id1++) {
    
    // Go over image cols (width)
    for (int get_global_id0 = 0; get_global_id0 < IMAGE_W; get_global_id0++) {
      
      int rowOffset = get_global_id1 * IMAGE_W * 4;
      int my = 4 * get_global_id0 + rowOffset;
      
      if ( get_global_id0 < HALF_FILTER_SIZE || 
	   get_global_id0 > IMAGE_W - HALF_FILTER_SIZE - 1 || 
	   get_global_id1 < HALF_FILTER_SIZE ||
	   get_global_id1 > IMAGE_H - HALF_FILTER_SIZE - 1
	   )
	{
	  continue;
	}
      
      // perform convolution
      int fIndex = 0;
      output[my] = 0.0;
      output[my+1] = 0.0;
      output[my+2] = 0.0;
      output[my+3] = 0.0;
      
      for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
	{
	  int curRow = my + r * (IMAGE_W * 4);
	  for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
	    {
	      int offset = c * 4;
	      
	      output[ my   ] += input[ curRow + offset   ] * filter[ fIndex   ]; 
	      output[ my+1 ] += input[ curRow + offset+1 ] * filter[ fIndex+1 ];
	      output[ my+2 ] += input[ curRow + offset+2 ] * filter[ fIndex+2 ]; 
	      output[ my+3 ] += input[ curRow + offset+3 ] * filter[ fIndex+3 ];
	      
	      fIndex += 4;
	      
	    }
	}
    }
  }
}

  cl_platform_id platform;
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;


// Host code
int main(int argc, char** argv)
{
  // Talk to the OpenCL runtime layer, make sure it is there
  clsystem clsys;
  cl_init(&clsys);
 //printf("Done allocating OpenCL\n");

  // Overall data structures for running 4 different kernels
  convolve RunConvolveBase, RunConvolveFloat4, RunConvolveLocalsize8by8, RunConvolveLocalsize16by16;

  // OpenCL system information

  // Allocate 
  float *imageIn = AllocateImage(IMAGE_W,  IMAGE_H, RGBA_SIZE); 
  float *imageOut = AllocateImage(IMAGE_W, IMAGE_H, RGBA_SIZE); 
  float *filter = AllocateFilter(FILTER_W, RGBA_SIZE);
 
  // Parms for base
  RunConvolveBase.Width = IMAGE_W;
  RunConvolveBase.Height = IMAGE_H;
  RunConvolveBase.filterSize = FILTER_W;
  RunConvolveBase.openclKernelFile = "Kernels/1_base.cl";
  RunConvolveBase.imageIn = imageIn;
  RunConvolveBase.imageOut = imageOut;
  RunConvolveBase.filter = filter;
  RunConvolveBase.localsize = 8;          // Not used, set to default

  // Parms for float4
  RunConvolveFloat4.Width = IMAGE_W;
  RunConvolveFloat4.Height = IMAGE_H;
  RunConvolveFloat4.filterSize = FILTER_W;
  RunConvolveFloat4.openclKernelFile = "Kernels/2_float4.cl";
  RunConvolveFloat4.imageIn = imageIn;
  RunConvolveFloat4.imageOut = imageOut;
  RunConvolveFloat4.filter = filter;
  RunConvolveFloat4.localsize = 8;        // Not used, set to default

  // Parms for float4 + local 8x8
  RunConvolveLocalsize8by8.Width = IMAGE_W;
  RunConvolveLocalsize8by8.Height = IMAGE_H;
  RunConvolveLocalsize8by8.filterSize = FILTER_W;
  RunConvolveLocalsize8by8.openclKernelFile = "Kernels/3_local_8x8.cl";
  RunConvolveLocalsize8by8.imageIn = imageIn;
  RunConvolveLocalsize8by8.imageOut = imageOut;
  RunConvolveLocalsize8by8.filter = filter;
  RunConvolveLocalsize8by8.localsize = 8;       

  // Parms for float4 + local 16x16
  RunConvolveLocalsize16by16.Width = IMAGE_W;
  RunConvolveLocalsize16by16.Height = IMAGE_H;
  RunConvolveLocalsize16by16.filterSize = FILTER_W;
  RunConvolveLocalsize16by16.openclKernelFile = "Kernels/4_local_16x16.cl";
  RunConvolveLocalsize16by16.imageIn = imageIn;
  RunConvolveLocalsize16by16.imageOut = imageOut;
  RunConvolveLocalsize16by16.filter = filter;
  RunConvolveLocalsize16by16.localsize = 16;


  // Timing support
  unsigned long long CPUStart, CPUStop, openCL_baseStart, openCL_baseStop, openCL_float4Start, openCL_float4Stop, openCL_float4_local8_Start, openCL_float4_local8_Stop, openCL_float4_local16_Start, openCL_float4_local16_Stop;

  // Run CPU
  CPUStart = rdtsc();
  CPU_Convolve(imageIn, imageOut, filter);
  CPUStop = rdtsc();
  printf("CPU:%llu\n",(CPUStop-CPUStart));

  // Run OpenCL : base
  openCL_baseStart = rdtsc();
  convolve_go(&RunConvolveBase, &clsys);
  openCL_baseStop = rdtsc();
  printf("openCL_base:%llu\n",(openCL_baseStop-openCL_baseStart));

  // Run OpenCL : float4
  openCL_float4Start = rdtsc();
  convolve_go(&RunConvolveFloat4, &clsys);
  openCL_float4Stop = rdtsc();
  printf("openCL_float4:%llu\n",(openCL_float4Stop-openCL_float4Start));

  // Run OpenCL : float4 w/ local size 8x8
  openCL_float4_local8_Start = rdtsc();
  convolve_go(&RunConvolveLocalsize8by8, &clsys);
  openCL_float4_local8_Stop = rdtsc();
  printf("openCL_float_local8:%llu\n",(openCL_float4_local8_Stop - openCL_float4_local8_Start));

  // Run OpenCL : float4 w/ local size 16x16
  openCL_float4_local16_Start = rdtsc();
  convolve_go(&RunConvolveLocalsize16by16, &clsys);
  openCL_float4_local16_Stop = rdtsc();
  printf("openCL_float_local16:%llu\n",(openCL_float4_local16_Stop - openCL_float4_local16_Start));


  // Close down OpenCL
  cl_close(&clsys);
  
}



