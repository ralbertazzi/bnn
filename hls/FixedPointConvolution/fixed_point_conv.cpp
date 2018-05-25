#include "fixed_point_conv.h"
#include <stdio.h>


f_type sign_and_accumulate(f_type pixels[K][K], ap_uint<K2> weights)
{
#pragma HLS INLINE

	f_type acc = 0;

	for(int i=0; i < K; i++)
	{
		f_type row_accumulator = 0;
		for(int j=0; j < K; j++)
			if(weights[i*K +j] == 0)
				row_accumulator -= pixels[i][j];
			else
				row_accumulator += pixels[i][j];

		acc += row_accumulator;
	}

	return acc;
}

void fixed_point_convolution(hls::stream<f_type> &input_stream, hls::stream<out_type> &output_stream,
		bit weight_interrupt, hls::stream<u64> &weight_channel)
{

#pragma HLS INTERFACE axis port=weight_channel
#pragma HLS INTERFACE s_axilite port=weight_interrupt
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE axis port=input_stream

	static ap_uint<K2> weights[FAN_OUT];
	static f_type thresholds[FAN_OUT];


#if P_OUT > 1 && P_OUT < FAN_OUT
	// Set array partition to cyclic factor=P_OUT
	#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=4 dim=1
	#pragma HLS ARRAY_PARTITION variable=thresholds cyclic factor=4 dim=1
#else
	#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
	#pragma HLS ARRAY_PARTITION variable=thresholds complete dim=1
#endif



	//line buffer
	static f_type line_buffer[K - 1][WIDTH_IMG];
	//#pragma HLS RESOURCE variable=line_buffer core=RAM_2P_BRAM
	#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

	static f_type line_buffer_temp [K - 1];
	#pragma HLS ARRAY_PARTITION variable=line_buffer_temp complete dim=0

	//processing window
	static f_type window[K][K];
	#pragma HLS ARRAY_PARTITION variable=window complete dim=0


	if(weight_interrupt)
	{
		weight_interrupt = 0;

		for(int fo=0; fo < FAN_OUT; fo++)
			weights[fo] = (ap_uint<K2>) weight_channel.read(); // Only the K2 LSBs are used

		for(int fo=0; fo < FAN_OUT; fo++)
			thresholds[fo] = (f_type) weight_channel.read();
	}

	out_type out_temp;


	Loop_row: for(int row = 0; row < HEIGHT_IMG; row++)
		Loop_col: for(int col = 0; col < WIDTH_IMG; col++)
		{

#if P_OUT < FAN_OUT
			Loop_p_out: for(int p_out_it=0; p_out_it < FAN_OUT / P_OUT; p_out_it++)
			{

	#pragma HLS PIPELINE II=2

				if(p_out_it == 0)
				{
					// shift columns of processing window
					loop_shift_window: for(int ii = 0; ii < K; ii++)
						for(int jj = 0; jj < K-1; jj++)
							window[ii][jj] = window[ii][jj+1];

					//line_buffer_temp
					loop_line_buffer_temp: for(int ii = 0; ii < K - 1; ii++)
						line_buffer_temp[ii] = line_buffer[ii][col];


					// copy K_H - 1 values from line_buffer to processing window
					loop_update_window: for(int ii = 0; ii < K - 1; ii++)
						window[ii][K - 1] = line_buffer_temp[ii];


					//shift row of line buffer
					loop_shift_line_buffer: for(int ii = 0; ii < K-2; ii++)
						line_buffer[ii][col] = line_buffer_temp[ii+1];


					f_type in_temp = input_stream.read();
					window[K-1][K-1]  = in_temp;
					line_buffer[K-2][col] = in_temp;
				}

				fmaps_out_loop: for(int po = 0; po < P_OUT; po++)
				{
					int out_idx = p_out_it * P_OUT + po;

					f_type accumulator = sign_and_accumulate(window, weights[out_idx]);

					if (accumulator >= thresholds[out_idx])
						out_temp[out_idx] = 1;
					else
						out_temp[out_idx] = 0;
				}

				if(p_out_it == FAN_OUT / P_OUT - 1)
				{
					if (row >= K - 1 && col >= K - 1)
					{
						output_stream.write(out_temp);
					}
				}

			}
#else

#pragma HLS PIPELINE II=1

			// shift columns of processing window
			loop_shift_window: for(int ii = 0; ii < K; ii++)
				for(int jj = 0; jj < K-1; jj++)
					window[ii][jj] = window[ii][jj+1];

			//line_buffer_temp
			loop_line_buffer_temp: for(int ii = 0; ii < K - 1; ii++)
				line_buffer_temp[ii] = line_buffer[ii][col];


			// copy K_H - 1 values from line_buffer to processing window
			loop_update_window: for(int ii = 0; ii < K - 1; ii++)
				window[ii][K - 1] = line_buffer_temp[ii];


			//shift row of line buffer
			loop_shift_line_buffer: for(int ii = 0; ii < K-2; ii++)
				line_buffer[ii][col] = line_buffer_temp[ii+1];


			f_type in_temp = input_stream.read();
			window[K-1][K-1]  = in_temp;
			line_buffer[K-2][col] = in_temp;


			fmaps_out_loop: for(int fo = 0; fo < FAN_OUT; fo++)
			{
				f_type accumulator = xnor_and_popcount_fixed(window, weights[fo]);

				if (accumulator >= thresholds[fo])
					out_temp[fo] = 1;
				else
					out_temp[fo] = 0;
			}


			if (row >= K - 1 && col >= K - 1)
			{
				output_stream.write(out_temp);
			}
#endif
		}
}
