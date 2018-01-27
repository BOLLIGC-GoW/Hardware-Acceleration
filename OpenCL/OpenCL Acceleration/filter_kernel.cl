#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_

__kernel void sobel(__global unsigned char *bufferImageIn, 
                    __global unsigned char *bufferImageOut, 
                    int Height, 
                    int Width, 
                    int Radius)
{
	uint x  = get_global_id(0);
	uint y  = get_global_id(1);
	Width   = get_global_size(0);
	Height  = get_global_size(1);
	
	
    if ((x <= 0+Radius) || (x >= Width-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

    if ((y <= 0+Radius) || (y >= Height-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

    float sobel_x[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  	float sobel_y[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  	float sumX = 0.0;
    float sumY = 0.0;
	
	if( x >= 1 && x < (Width-1) && y >= 1 && y < Height - 1 )
		{
			sumX += (float)(bufferImageIn[(x - 1) + (y - 1) * Width])* (sobel_x[0]);
			sumY += (float)(bufferImageIn[(x - 1) + (y - 1) * Width])* (sobel_y[0]);

			sumX += (float)(bufferImageIn[x + (y - 1) * Width])* (sobel_x[1]);
			sumY += (float)(bufferImageIn[x + (y - 1) * Width])* (sobel_y[1]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 1) * Width])* (sobel_x[2]);
			sumY += (float)(bufferImageIn[(x + 1) + (y - 1) * Width])* (sobel_y[2]);

			sumX += (float)(bufferImageIn[(x - 1) + y * Width])* (sobel_x[3]);
			sumY += (float)(bufferImageIn[(x - 1) + y * Width])* (sobel_y[3]);

			sumX += (float)(bufferImageIn[x + y * Width]) *  (sobel_x[4]);
			sumY += (float)(bufferImageIn[x + y * Width]) *  (sobel_y[4]);

			sumX += (float)(bufferImageIn[(x + 1) + y * Width])* (sobel_x[5]);
			sumY += (float)(bufferImageIn[(x + 1) + y * Width])* (sobel_y[5]);

			sumX += (float)(bufferImageIn[(x - 1) + (y + 1) * Width])* (sobel_x[6]);
			sumY += (float)(bufferImageIn[(x - 1) + (y + 1) * Width])* (sobel_y[6]);

			sumX += (float)(bufferImageIn[x + (y + 1) * Width])* (sobel_x[7]);
			sumY += (float)(bufferImageIn[x + (y + 1) * Width])* (sobel_y[7]);

			sumX += (float)(bufferImageIn[(x + 1) + (y + 1) * Width])* (sobel_x[8]);
			sumY += (float)(bufferImageIn[(x + 1) + (y + 1) * Width])* (sobel_y[8]);

			bufferImageOut[x + y * Width] = (char)(fabs(sumX) + fabs(sumY)) > 70.0 ? 255.0 : 0.0;
		}  
}

__kernel void sobel5x5(__global unsigned char *bufferImageIn, 
                    __global unsigned char *bufferImageOut, 
                    int Height, 
                    int Width, 
                    int Radius)
{
	uint x  = get_global_id(0);
	uint y  = get_global_id(1);
	Width   = get_global_size(0);
	Height  = get_global_size(1);
	
	
    if ((x <= 0+2) || (x >= Width-2))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

    if ((y <= 0+2) || (y >= Height-2))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

    float sobel_x[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};
  	float sobel_y[25] = {1,2,0,-2,-1,4,8,0,-8,-4,6,12,0,-12,6,4,8,0,-8,-4,1,2,0,-2,-1};
  	float sumX = 0.0;
    float sumY = 0.0;
	
	if( x >= 1 && x < (Width-1) && y >= 1 && y < Height - 1 )
		{

			sumX += (float)(bufferImageIn[(x - 2) + (y - 2) * Width])* (sobel_x[0]);
			sumY += (float)(bufferImageIn[(x - 2) + (y - 2) * Width])* (sobel_y[0]);

			sumX += (float)(bufferImageIn[(x - 1) + (y - 2) * Width])* (sobel_x[1]);
			sumY += (float)(bufferImageIn[(x - 1) + (y - 2) * Width])* (sobel_y[1]);

			sumX += (float)(bufferImageIn[x + (y - 2) * Width])* (sobel_x[2]);
			sumY += (float)(bufferImageIn[x + (y - 2) * Width])* (sobel_y[2]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 2) * Width])* (sobel_x[3]);
			sumY += (float)(bufferImageIn[(x + 1) + (y - 2) * Width])* (sobel_y[3]);

			sumX += (float)(bufferImageIn[(x + 2) + (y - 2) * Width])* (sobel_x[4]);
			sumY += (float)(bufferImageIn[(x + 2) + (y - 2) * Width])* (sobel_y[4]);

			sumX += (float)(bufferImageIn[(x - 2) + (y - 1) * Width])* (sobel_x[5]);
			sumY += (float)(bufferImageIn[(x - 2) + (y - 1) * Width])* (sobel_y[5]);

			sumX += (float)(bufferImageIn[(x - 1) + (y - 1) * Width])* (sobel_x[6]);
			sumY += (float)(bufferImageIn[(x - 1) + (y - 1) * Width])* (sobel_y[6]);

			sumX += (float)(bufferImageIn[x + (y - 1) * Width])* (sobel_x[7]);
			sumY += (float)(bufferImageIn[x + (y - 1) * Width])* (sobel_y[7]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 1) * Width])* (sobel_x[8]);
			sumY += (float)(bufferImageIn[(x + 1) + (y - 1) * Width])* (sobel_y[8]);

			sumX += (float)(bufferImageIn[(x + 2) + (y - 1) * Width])* (sobel_x[9]);
			sumY += (float)(bufferImageIn[(x + 2) + (y - 1) * Width])* (sobel_y[9]);

			sumX += (float)(bufferImageIn[(x - 2) + (y) * Width])* (sobel_x[10]);
			sumY += (float)(bufferImageIn[(x - 2) + (y) * Width])* (sobel_y[10]);

			sumX += (float)(bufferImageIn[(x - 1) + (y) * Width])* (sobel_x[11]);
			sumY += (float)(bufferImageIn[(x - 1) + (y) * Width])* (sobel_y[11]);

			sumX += (float)(bufferImageIn[x + (y) * Width])* (sobel_x[12]);
			sumY += (float)(bufferImageIn[x + (y) * Width])* (sobel_y[12]);

			sumX += (float)(bufferImageIn[(x + 1) + (y) * Width])* (sobel_x[13]);
			sumY += (float)(bufferImageIn[(x + 1) + (y) * Width])* (sobel_y[13]);

			sumX += (float)(bufferImageIn[(x + 2) + (y) * Width])* (sobel_x[14]);
			sumY += (float)(bufferImageIn[(x + 2) + (y) * Width])* (sobel_y[14]);

			sumX += (float)(bufferImageIn[(x - 2) + (y + 1) * Width])* (sobel_x[15]);
			sumY += (float)(bufferImageIn[(x - 2) + (y + 1) * Width])* (sobel_y[15]);

			sumX += (float)(bufferImageIn[(x - 1) + (y + 1) * Width])* (sobel_x[16]);
			sumY += (float)(bufferImageIn[(x - 1) + (y + 1) * Width])* (sobel_y[16]);

			sumX += (float)(bufferImageIn[x + (y + 1) * Width])* (sobel_x[17]);
			sumY += (float)(bufferImageIn[x + (y + 1) * Width])* (sobel_y[17]);

			sumX += (float)(bufferImageIn[(x + 1) + (y + 1) * Width])* (sobel_x[18]);
			sumY += (float)(bufferImageIn[(x + 1) + (y + 1) * Width])* (sobel_y[18]);

			sumX += (float)(bufferImageIn[(x + 2) + (y + 1) * Width])* (sobel_x[19]);
			sumY += (float)(bufferImageIn[(x + 2) + (y + 1) * Width])* (sobel_y[19]);

			sumX += (float)(bufferImageIn[(x - 2) + (y + 2) * Width])* (sobel_x[20]);
			sumY += (float)(bufferImageIn[(x - 2) + (y + 2) * Width])* (sobel_y[20]);

			sumX += (float)(bufferImageIn[(x - 1) + (y + 2) * Width])* (sobel_x[21]);
			sumY += (float)(bufferImageIn[(x - 1) + (y + 2) * Width])* (sobel_y[21]);

			sumX += (float)(bufferImageIn[x + (y + 2) * Width])* (sobel_x[22]);
			sumY += (float)(bufferImageIn[x + (y + 2) * Width])* (sobel_y[22]);

			sumX += (float)(bufferImageIn[(x + 1) + (y + 2) * Width])* (sobel_x[23]);
			sumY += (float)(bufferImageIn[(x + 1) + (y + 2) * Width])* (sobel_y[23]);

			sumX += (float)(bufferImageIn[(x + 2) + (y + 2) * Width])* (sobel_x[24]);
			sumY += (float)(bufferImageIn[(x + 2) + (y + 2) * Width])* (sobel_y[24]);



			bufferImageOut[x + y * Width] = (char)(fabs(sumX) + fabs(sumY)) > 70.0 ? 255.0 : 0.0;
		}  
}



__kernel void average(__global unsigned char *bufferImageIn, 
                    __global unsigned char *bufferImageOut, 
                    int Height, 
                    int Width, 
                    int Radius)
{
	uint x  = get_global_id(0);
	uint y  = get_global_id(1);
	Width   = get_global_size(0);
	Height  = get_global_size(1);
	
	
    if ((x <= 0+Radius) || (x >= Width-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

    if ((y <= 0+Radius) || (y >= Height-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

  	float sumX = 0.0;
	
	if( x >= 1 && x < (Width-1) && y >= 1 && y < Height - 1 )
		{
			sumX += (float)(bufferImageIn[(x - 1) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[x + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + y * Width]);

			sumX += (float)(bufferImageIn[x + y * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + y * Width]);

			sumX += (float)(bufferImageIn[ (x - 1) + (y + 1) ]);

			sumX += (float)(bufferImageIn [ x + (y + 1) * Width ] );

			sumX += (float)(bufferImageIn[(x + 1) + (y + 1) * Width]);

			bufferImageOut[x + y * Width] = (char)(sumX / 9.0);
		}  
}

__kernel void average5x5(__global unsigned char *bufferImageIn, 
                    __global unsigned char *bufferImageOut, 
                    int Height, 
                    int Width, 
                    int Radius)
{
	uint x  = get_global_id(0);
	uint y  = get_global_id(1);
	Width   = get_global_size(0);
	Height  = get_global_size(1);
	
	
    if ((x <= 0+Radius) || (x >= Width-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

    if ((y <= 0+Radius) || (y >= Height-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

  	float sumX = 0.0;
	
	if( x >= 1 && x < (Width-1) && y >= 1 && y < Height - 1 )
		{
			sumX += (float)(bufferImageIn[(x - 2) + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[x + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[(x - 2) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[x + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 2) + (y) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y) * Width]);

			sumX += (float)(bufferImageIn[x + (y) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y) * Width]);

			sumX += (float)(bufferImageIn[(x - 2) + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[x + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 2) + (y + 2) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y + 2) * Width]);

			sumX += (float)(bufferImageIn[x + (y + 2) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y + 2) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y + 2) * Width]);

			bufferImageOut[x + y * Width] = (char)(sumX / 25.0);
		}  
}

__kernel void boost(__global unsigned char *bufferImageIn, 
                    __global unsigned char *bufferImageOut, 
                    int Height, 
                    int Width, 
                    int Radius)
{
	uint x  = get_global_id(0);
	uint y  = get_global_id(1);
	Width   = get_global_size(0);
	Height  = get_global_size(1);
	
	
    if ((x <= 0+Radius) || (x >= Width-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

    if ((y <= 0+Radius) || (y >= Height-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

  	float sumX = 0.0;
	
	if( x >= 1 && x < (Width-1) && y >= 1 && y < Height - 1 )
		{
			sumX += (float)(bufferImageIn[(x - 1) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[x + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + y * Width]);

			sumX += (float)(bufferImageIn[x + y * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + y * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y + 1)]);

			sumX += (float)(bufferImageIn[x + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y + 1) * Width]);

			bufferImageOut[x + y * Width] = (char)(((sumX / 9.0) - (float)(bufferImageIn[x + y * Width])) * 10);
		}  
}

__kernel void boost5x5(__global unsigned char *bufferImageIn, 
                    __global unsigned char *bufferImageOut, 
                    int Height, 
                    int Width, 
                    int Radius)
{
	uint x  = get_global_id(0);
	uint y  = get_global_id(1);
	Width   = get_global_size(0);
	Height  = get_global_size(1);
	
	
    if ((x <= 0+Radius) || (x >= Width-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

    if ((y <= 0+Radius) || (y >= Height-Radius))  {
       bufferImageOut[y*Width + x] = 0;
       return;
    }

  	float sumX = 0.0;
	
	if( x >= 1 && x < (Width-1) && y >= 1 && y < Height - 1 )
		{
			sumX += (float)(bufferImageIn[(x - 2) + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[x + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y - 2) * Width]);

			sumX += (float)(bufferImageIn[(x - 2) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[x + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y - 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 2) + (y) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y) * Width]);

			sumX += (float)(bufferImageIn[x + (y) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y) * Width]);

			sumX += (float)(bufferImageIn[(x - 2) + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[x + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y + 1) * Width]);

			sumX += (float)(bufferImageIn[(x - 2) + (y + 2) * Width]);

			sumX += (float)(bufferImageIn[(x - 1) + (y + 2) * Width]);

			sumX += (float)(bufferImageIn[x + (y + 2) * Width]);

			sumX += (float)(bufferImageIn[(x + 1) + (y + 2) * Width]);

			sumX += (float)(bufferImageIn[(x + 2) + (y + 2) * Width]);

			bufferImageOut[x + y * Width] = (char)(((sumX / 25.0) - (float)(bufferImageIn[x + y * Width])) * 10);
		}  
}

#endif // _FILTER_KERNEL_H_




