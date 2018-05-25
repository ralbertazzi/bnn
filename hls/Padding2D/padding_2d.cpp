#include "padding_2d.h"

void padding_2d(hls::stream<in_type> &input_stream, hls::stream<in_type> &output_stream)
{
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE axis port=input_stream

	in_type default_value;
	loop_init: for(int i=0; i < FMAPS; i++)
	{
		#pragma HLS UNROLL
		default_value.set(i, PAD_VALUE);
	}

/*
 * First implementation: working but for some reasons I get problems enforcing II=1
 */

	for(int row=0; row < IMG_HEIGHT + 2*PAD; row++)
		for(int col=0; col < IMG_WIDTH + 2*PAD; col++)
		{
#pragma HLS PIPELINE II=2
			if(row >= PAD and row < IMG_HEIGHT + PAD and col >= PAD and col < IMG_WIDTH + PAD)
			{
				in_type val = input_stream.read();
				output_stream.write(val);
			}
			else
			{
				output_stream.write(default_value);
			}

		}

/*
 * Second implementation: row padding is now explicit (only variable col needs to be compared).
 * Allows to enforce II=1
 */

//	loop_upper_pad_rows: for(int p=0; p < PAD; p++)
//		for(int col=0; col < IMG_WIDTH + 2*PAD; col++)
//			output_stream.write(default_value);
//
//	loop_image_rows: for(int row=0; row < IMG_HEIGHT; row++)
//		for(int col=0; col < IMG_WIDTH + 2*PAD; col++)
//		{
//#pragma HLS PIPELINE II=1
//			if(col >= PAD and col < (IMG_WIDTH + PAD))
//			{
//				in_type val = input_stream.read();
//				output_stream.write(val);
//			}
//			else
//			{
//				output_stream.write(default_value);
//			}
//		}
//
//	loop_lower_pad_rows: for(int p=0; p < PAD; p++)
//		for(int col=0; col < IMG_WIDTH + 2*PAD; col++)
//			output_stream.write(default_value);

}
