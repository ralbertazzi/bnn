#include "binary_conv.h"
#include <stdio.h>

ap_uint<K2_BITS> xnor_and_popcount(ap_uint<K2> pixels, ap_uint<K2> weights)
{
#pragma HLS INLINE

	ap_uint<K2_BITS> acc = 0;


	ap_uint<K2> c;
	for (int k=0; k < K2; k++)
		c[k] = (pixels[k] == weights[k]);

	for(int k=0; k < K; k++)
	{
		ap_uint<K_BITS> row_accumulator = 0;
		for(int h=0; h < K; h++)
			if(c[k*K + h] == true)
				row_accumulator++;

		acc += row_accumulator;
	}

	return acc;
}


void binary_convolution(hls::stream<in_type> &input_stream, hls::stream<out_type> &output_stream,
		bit weight_interrupt, hls::stream<u64> &weight_channel)
{

#pragma HLS INTERFACE axis port=weight_channel
#pragma HLS INTERFACE s_axilite port=weight_interrupt
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE axis port=input_stream

#ifndef OPTIMIZED
	static ap_uint<K2> weights[FMAPS_OUT / P_OUT][FMAPS_IN / P_IN][P_OUT][P_IN];

	#pragma HLS ARRAY_RESHAPE variable=weights block factor=4 dim=4
	#pragma HLS ARRAY_PARTITION variable=weights complete dim=3
	#pragma HLS ARRAY_PARTITION variable=weights complete dim=4
#else
	static ap_uint<K2> weights[FMAPS_OUT][FMAPS_IN];

	#pragma HLS ARRAY_RESHAPE variable=weights block factor=4 dim=2
	#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#endif

#pragma HLS RESOURCE variable=weights core=RAM_1P_BRAM

	static acc_type thresholds[FMAPS_OUT];

#ifndef OPTIMIZED
	#if P_OUT == FMAPS_OUT
		#pragma HLS ARRAY_PARTITION variable=thresholds complete dim=1
	#elif P_OUT > 1 && P_OUT < FMAPS_OUT
		//Set to cyclic=P_OUT
		#pragma HLS ARRAY_PARTITION variable=thresholds cyclic factor=2 dim=1
	#endif
#endif

	//line buffer
	static in_type line_buffer[K - 1][WIDTH_IMG];
	// #pragma HLS RESOURCE variable=line_buffer core=RAM_2P_BRAM
	#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

	//processing window
	static in_type window[K][K];
	#pragma HLS ARRAY_PARTITION variable=window complete dim=0

#pragma HLS DEPENDENCE variable=line_buffer


	if(weight_interrupt)
	{
		weight_interrupt = 0;

		/*
		 * Read weights
		 */
#ifndef OPTIMIZED
		for(int fo=0; fo < FMAPS_OUT / P_OUT; fo++)
			for(int fi=0; fi < FMAPS_IN / P_IN; fi++)
				for(int po=0; po < P_OUT; po++)
					for(int pi=0; pi < P_IN; pi++)
						weights[fo][fi][po][pi] = (ap_uint<K2>) weight_channel.read(); // Only the K2 LSBs are used
#else
		for(int fo=0; fo < FMAPS_OUT / P_OUT; fo++)
				for(int fi=0; fi < FMAPS_IN; fi++)
					weights[fo][fi] = (ap_uint<K2>) weight_channel.read(); // Only the K2 LSBs are used
#endif

		/*
		 * Read thresholds
		 */
		for(int fo=0; fo < FMAPS_OUT; fo++)
			thresholds[fo] = (acc_type) weight_channel.read(); // Only the ACC_BITS LSBs are used
	}


	out_type out_temp;

	Loop_row: for(int row = 0; row < HEIGHT_IMG; row++){
		Loop_col: for(int col = 0; col < WIDTH_IMG; col++)
		{

#ifndef OPTIMIZED

			loop_p_out: for(int p_out_it = 0; p_out_it < FMAPS_OUT / P_OUT; p_out_it++)
			{
				loop_p_in: for(int p_in_it = 0; p_in_it < FMAPS_IN / P_IN; p_in_it++)
				{

#pragma HLS PIPELINE II=2

					if (p_out_it == 0 && p_in_it == 0)
					{
						// shift columns of processing window
						loop_shift_window: for(int ii = 0; ii < K; ii++)
							for(int jj = 0; jj < K-1; jj++)
								window[ii][jj] = window[ii][jj+1];


						//line_buffer_temp
						loop_line_buffer_temp: for(int ii = 0; ii < K - 1; ii++)
							window[ii][K - 1] = line_buffer[ii][col];

						//shift row of line buffer
						loop_shift_line_buffer: for(int ii = 0; ii < K-2; ii++)
							line_buffer[ii][col] = line_buffer[ii+1][col];

						in_type in_temp = input_stream.read();
						window[K-1][K-1]  = in_temp;
						line_buffer[K-2][col] = in_temp;
					}


					ap_uint<K2> i_data[P_IN];
					acc_type accumulators[P_OUT];

					if (p_in_it == 0)
					{
						for(int po = 0; po < P_OUT; po++)
							accumulators[po] = 0;
					}

					/*
					 * Initialize i_data for convolution computation
					 * (we basically reshape the window)
					 */

					for(int pi = 0; pi < P_IN; pi++)
						for(int i=0; i < K; i++)
							for(int j=0; j < K; j++)
								i_data[pi][i*K + j] = window[i][j][p_in_it * P_IN + pi];

					/*
					 * Compute P_IN * P_OUT convolutions
					 */

					fmaps_out_loop: for(int po = 0; po < P_OUT; po++)
					{
						fmaps_in_loop: for(int pi=0; pi < P_IN; pi++)
						{
							accumulators[po] += xnor_and_popcount(i_data[pi], weights[p_out_it][p_in_it][po][pi]);
						}
					}


					if (p_in_it == FMAPS_IN / P_IN - 1)
					{

						loop_thresholds: for(int po = 0; po < P_OUT; po++)
						{
							int fmaps_out_idx = p_out_it*P_OUT + po;
							if (accumulators[po] >= thresholds[fmaps_out_idx])
								out_temp[fmaps_out_idx] =  1;
							else
								out_temp[fmaps_out_idx] = 0;
						}


					}

					if (p_in_it == FMAPS_IN / P_IN - 1 && p_out_it == FMAPS_OUT / P_OUT - 1)
					{
						if (row >= K - 1 && col >= K - 1)
							output_stream.write(out_temp);
					}


				}

			}

#else

				loop_p_out: for(int po = 0; po < FMAPS_OUT; po++)
				{

#pragma HLS PIPELINE II=2

					if (po == 0)
					{
						// shift columns of processing window
						loop_shift_window: for(int ii = 0; ii < K; ii++)
							for(int jj = 0; jj < K-1; jj++)
								window[ii][jj] = window[ii][jj+1];


						//line_buffer_temp
						loop_line_buffer_temp: for(int ii = 0; ii < K - 1; ii++)
							window[ii][K - 1] = line_buffer[ii][col];

						//shift row of line buffer
						loop_shift_line_buffer: for(int ii = 0; ii < K-2; ii++)
							line_buffer[ii][col] = line_buffer[ii+1][col];

						in_type in_temp = input_stream.read();
						window[K-1][K-1]  = in_temp;
						line_buffer[K-2][col] = in_temp;
					}

					if (row >= K - 1 && col >= K - 1)
					{
						ap_uint<K2> i_data[FMAPS_IN];
						acc_type accumulator = 0;

						/*
						 * Initialize i_data for convolution computation
						 * (we basically reshape the window)
						 */

						for(int fi = 0; fi < FMAPS_IN; fi++)
							for(int i=0; i < K; i++)
								for(int j=0; j < K; j++)
									i_data[fi][i*K + j] = window[i][j][fi];

						/*
						 * Compute P_IN * P_OUT convolutions
						 */

						fmaps_in_loop: for(int fi=0; fi < FMAPS_IN; fi++)
						{
							accumulator += xnor_and_popcount(i_data[fi], weights[po][fi]);
						}

						if (accumulator >= thresholds[po])
							out_temp[po] =  1;
						else
							out_temp[po] = 0;

						if (po == FMAPS_OUT - 1)
						{
							output_stream.write(out_temp);
						}
					}


				}

#endif

		} //loop columns
	}//loop rows
}
