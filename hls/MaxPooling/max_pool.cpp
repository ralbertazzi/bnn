#include "max_pool.h"
#include <stdio.h>

in_type compute_max_pool(in_type window[K][K])
{
	/*
	 * Since we are dealing with binary images, performing max-pooling is
	 * equivalent to computing the OR between the bits inside the image.
	 * Since we don't have just one image, but multiple feature maps, we
	 * are going to compute bitwise ORs.
	 */

	/*
	 * This is the simplest possible implementation, but it works!
	 * I tried a second version were I was defining one 'OR-accumulator' per row,
	 * and then making the OR of the OR-accumulators, but the resource
	 * consumption was exactly the same (even for higher sizes of K).
	 */

	#pragma HLS INLINE

	in_type or_total = 0;
	for(int i = 0; i < K; i++)
		for(int j = 0; j < K; j++)
		{
			or_total = or_total | window[i][j];
		}

	return or_total;

}

void max_pool(hls::stream<in_type> &in_img, hls::stream<in_type> &out_img)
{

#pragma HLS INTERFACE axis port=out_img
#pragma HLS INTERFACE axis port=in_img

//line buffer
static in_type line_buffer[K - 1][WIDTH_IMG];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
static in_type line_buffer_temp [K - 1];
#pragma HLS ARRAY_PARTITION variable=line_buffer_temp complete dim=0

//processing window
static in_type window[K][K];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0


Loop_row: for(int row = 0; row < HEIGHT_IMG; row++){
	Loop_col: for(int col = 0; col < WIDTH_IMG; col++)

	{
		#pragma HLS PIPELINE II=1

		in_type in_temp = in_img.read();

		// shift columns of processing window
		for(int ii = 0; ii < K; ii++)
			for(int jj = 0; jj < K-1; jj++)
				window[ii][jj] = window[ii][jj+1];

		//line_buffer_temp
		for(int ii = 0; ii < K - 1; ii++)
			line_buffer_temp[ii] = line_buffer[ii][col];

		// copy K_H - 1 values from line_buffer to processing window
		for(int ii = 0; ii < K - 1; ii++)
			window[ii][K - 1] = line_buffer_temp[ii];

		//shift row of line buffer
		for(int ii = 0; ii < K-2; ii++)
			line_buffer[ii][col]= line_buffer_temp[ii+1];

		window[K-1][K-1] = in_temp;
		line_buffer[K-2][col] = in_temp;

		//output value only if we are in the correct position to compute the max
		if (row % K == K - 1 and col % K == K - 1)
		{
			in_type out_temp = compute_max_pool(window);
			out_img.write(out_temp);
		}

    } //loop columns
}//loop rows

}
