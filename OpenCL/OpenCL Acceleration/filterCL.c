// 
// Filters
//

// Includes: system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
// OpenCL 

// OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Includes: bmp and filter
#include "bmp.h"
#include "filter.h"

enum {SOBEL_FILTER=1, AVERAGE_FILTER, HIGH_BOOST_FILTER};

// Functions
void Cleanup(void);
void ParseFilterArguments(filter *, int, char**);


struct timeval timerStartCPU;

void StartTimer_CPU() {
  gettimeofday(&timerStartCPU, NULL);
}

double GetTimer_CPU() {
  struct timeval timerStopCPU, timerElapsedCPU;
  gettimeofday(&timerStopCPU, NULL);
  timersub(&timerStopCPU, &timerStartCPU, &timerElapsedCPU);
  return timerElapsedCPU.tv_sec*1000.0+timerElapsedCPU.tv_usec/1000.0;
}

void CPU_Boost3x3(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;

  // Default filter setup
  int FILTER_RADIUS = 1;

  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
  imageOut[i*width + j] = 0;
    }
  }

  startCol = 1;
  endCol = cols - 1;
  startRow = 1;
  endRow = rows - 1;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {

       // sum up the 9 values to calculate both the direction x direction
       int sumX = 0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             int Pixel = (imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel;
          }
  }     
      //Take the average of the sum of the values
       imageOut[i*width + j] = ((int)(sumX / 9)-(imageIn[0])*10);
    }
  }
}
void CPU_Boost5x5(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;

  // Default filter setup
  int FILTER_RADIUS = 2;


  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
  imageOut[i*width + j] = 0;
    }
  }

  startCol = 2;
  endCol = cols - 2;
  startRow = 2;
  endRow = rows - 2;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {

       // sum up the 25 values to calculate both the direction x direction
       int sumX = 0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             int Pixel = (imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel;
          }
  }     
      //Take the average of the sum of the values
       imageOut[i*width + j] = ((int)(sumX / 9)-(imageIn[0])*10);
    }
  }
}
void CPU_Average3x3(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;

  // Default filter setup
  int FILTER_RADIUS = 1;

  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
  imageOut[i*width + j] = 0;
    }
  }

  startCol = 1;
  endCol = cols - 1;
  startRow = 1;
  endRow = rows - 1;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {

       // sum up the 9 values to calculate both the direction x direction
       int sumX = 0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             int Pixel = (imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel;
          }
  }     
      //Take the average of the sum of the values
       imageOut[i*width + j] = sumX / 9;
    }
  }
}
void CPU_Average5x5(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;

  // Default filter setup
  int FILTER_RADIUS = 2;

  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
  imageOut[i*width + j] = 0;
    }
  }

  startCol = 2;
  endCol = cols - 2;
  startRow = 2;
  endRow = rows - 2;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {

       // sum up the 25 values to calculate both the direction x direction
       int sumX = 0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             int Pixel = ( imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel;
          }
  }     
      //Take the average of the sum of the values
       imageOut[i*width + j] = sumX / 9;
    }
  }
}
void CPU_Sobel3x3(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;
  char SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};

  // Default filter setup
  int EDGE_VALUE_THRESHOLD = 70;
  int FILTER_RADIUS = 1;
  int FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;

  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
	imageOut[i*width + j] = 0;
    }
  }

  startCol = 1;
  endCol = cols - 1;
  startRow = 1;
  endRow = rows - 1;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {

       // sum up the 9 values to calculate both the direction x and direction y
       int sumX = 0, sumY=0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             int Pixel = (imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel * SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
             sumY += Pixel * SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)];
          }
	}
       imageOut[i*width + j] = (unsigned char) ((abs(sumX) + abs(sumY)) > EDGE_VALUE_THRESHOLD ? 255 : 0);
    }
  }
}
void CPU_Sobel5x5(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;
  char SobelMatrix[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};

  // Default filter setup
  int EDGE_VALUE_THRESHOLD = 70;
  int FILTER_RADIUS = 2;
  int FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;

  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
  imageOut[i*width + j] = 0;
    }
  }

  startCol = 2;
  endCol = cols - 2;
  startRow = 2;
  endRow = rows - 2;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {

       // sum up the 9 values to calculate both the direction x and direction y
       int sumX = 0, sumY=0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             int Pixel = (imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel * SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
             sumY += Pixel * SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)];
          }
  }
       imageOut[i*width + j] = (unsigned char) ((abs(sumX) + abs(sumY)) > EDGE_VALUE_THRESHOLD ? 255 : 0);
    }
  }
}
// Host code
int main(int argc, char** argv)
{
    // Overall data structure
    filter RunFilter;

    // Parse command line arguments
    ParseFilterArguments(&RunFilter, argc, argv);

    struct bmp_header bmp;
    struct dib_header dib;
    unsigned char *palete = NULL;

    // Read bitmap file
    BitMapRead(RunFilter.bitmapInputFile, &RunFilter.imageIn, &bmp, &dib, &palete);
    RunFilter.imageOut = (unsigned char *)malloc(dib.image_size);

    // Image Details 
    printf("Image details: %d by %d = %d , imagesize = %d\n", dib.width, dib.height, dib.width * dib.height,dib.image_size);

    // CPU Sobel filter
    //Start CPU timer
    StartTimer_CPU();
    CPU_Sobel5x5(RunFilter.imageIn, RunFilter.imageOut, dib.width, dib.height);

        // Timing code stop
    double runtime = GetTimer_CPU();
    printf("Finished with CPU Kernel\n");
    printf("Computation Time: %f ms\n\n", runtime);
    BitMapWrite("CPU_sobel.bmp", RunFilter.imageOut, &bmp, &dib, palete);

    // Setup the filter's image kernel
    //char SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};
    //RunFilter.imageKernel = SobelMatrix;

    // Setup the filter to run
    RunFilter.Width = dib.width;  
    RunFilter.Height = dib.height;

    // Initialize OpenCL and run filter
    filter_go(&RunFilter);

    // Write output image   
    BitMapWrite(RunFilter.bitmapOutputFile, RunFilter.imageOut, &bmp, &dib, palete);
}

// Parse program arguments
void ParseFilterArguments(filter *self, int argc, char *argv[])
{
    // Default values
    self->kernelRadius =1;
    self->openclKernelFile = "filter_kernel.cl";
    self->filterName = "sobel";
    self->boost = 1;

    for (int i = 1; i < argc; i++) {

        if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-file") == 0) {
            i = i + 1;
            self->bitmapInputFile = argv[i];
        }
        else if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-out") == 0) {
            i = i + 1;
            self->bitmapOutputFile = argv[i];
        }
        else if (strcmp(argv[i], "--cl") == 0 || strcmp(argv[i], "-cl") == 0) {
            i = i + 1;
            self->openclKernelFile = argv[i];
        }
        else if (strcmp(argv[i], "--filter") == 0 || strcmp(argv[i], "-filter") == 0) {
            i = i + 1;
            self->filterName = argv[i];

            if (strcmp(self->filterName, "sobel") == 0)
                self->filterMode = SOBEL_FILTER;
            else if (strcmp(self->filterName, "average") == 0)
                self->filterMode = AVERAGE_FILTER;
            else if (strcmp(self->filterName, "boost") == 0)
                self->filterMode = HIGH_BOOST_FILTER;
        }
        else if (strcmp(argv[i], "--radius") == 0 || strcmp(argv[i], "-radius") == 0) {
            i = i + 1;
            self->kernelRadius = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--boost") == 0 || strcmp(argv[i], "-boost") == 0) {
            i = i + 1;
            self->boost = atoi(argv[i]);
        }
    }

    printf("Processing %s : %s -> %s\n", self->filterName, self->bitmapInputFile, self->bitmapOutputFile);
    printf("OpenCL file : %s \n", self->openclKernelFile);
    printf("Parameters : Radius %d BoostThreshold %d \n", self->kernelRadius, self->boost);
}



