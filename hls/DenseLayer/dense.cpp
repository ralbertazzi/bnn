#include "dense.h"

/*
 * A Dense layer has to compute a matrix multiplication between an input vector
 * of size (1, INPUT_DIM) and a weight matrix of size (INPUT_DIM, OUTPUT_DIM). The
 * size of the resulting vector will be (1, OUTPUT_DIM).
 * In this case all the operands (input vector and weights) are binary. The result
 * is a (unsigned) integer vector; each value of the following vector will be
 * thresholded by a different value (batch normalization + activation function).
 * The output vector is therefore a vector of binary values.
 *
 */

#if MODE == MODE_MATMUL_AND_THRESHOLD

void dense_layer(hls::stream<bit> &input_stream,
		hls::stream<bit> &output_stream,
		hls::stream<u64> &weights_channel,
		bit weights_interrupt)

#elif MODE == MODE_MATMUL_ONLY

void dense_layer(hls::stream<bit> &input_stream,
		hls::stream<acc_type> &output_stream,
		hls::stream<u64> &weights_channel,
		bit weights_interrupt)

#endif
{

#pragma HLS INTERFACE axis port=weights_channel
#pragma HLS INTERFACE s_axilite port=weights_interrupt
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE axis port=input_stream

	static output_type weights[INPUT_DIM];

#if MODE == MODE_MATMUL_AND_THRESHOLD
	static acc_type thresholds[OUTPUT_DIM];
#endif

// #pragma HLS ARRAY_PARTITION variable=weights complete dim=1

	if (weights_interrupt)
	{
		weights_interrupt = 0;
		output_type weight_row;
		u64 in_weight64;

		read_weights_loop: for(int in=0; in < INPUT_DIM; in++)
		{
			for (int o=0; o < OUTPUT_DIM; o++)
			{
				int o_mod_64 = o % 64;
				if (o_mod_64 == 0)
					in_weight64 = weights_channel.read();

				weight_row[o] = in_weight64[o_mod_64];
			}

			weights[in] = weight_row;
		}

#if MODE == MODE_MATMUL_AND_THRESHOLD
		read_threshold_loop: for(int nread=0; nread < OUTPUT_DIM; nread++)
		{
			thresholds[nread] = (acc_type) weights_channel.read(); // Use only LSBs
		}
#endif



	}

#if P_OUT == 0

	/*
	 * First implementation: in this implementation we keep a number of accumulators for each
	 * column of our weight matrix (OUTPUT_DIM). Each time a new input bit comes in, we compute
	 * the XNOR between the input bit and all the corresponding OUTPUT_DIM weights and eventually
	 * increase the accumulators in parallel. At the end we output each value sequentially.
	 *
	 * With this implementation we can achieve the maximum parallelism, at the cost of using
	 * a lot of resources.
	 */


	acc_type accumulators[OUTPUT_DIM];
#pragma HLS ARRAY_PARTITION variable=accumulators complete

	reset_accumulators: for(int out=0; out<OUTPUT_DIM; out++)
	{
#pragma HLS UNROLL
		accumulators[out] = 0;
	}

	// Main loop
	input_loop: for(int in=0; in < INPUT_DIM; in++)
	{

#pragma HLS PIPELINE II=1
		bit input_bit = input_stream.read();

		xnor_and_accumulate_loop: for(int out=0; out<OUTPUT_DIM; out++)
		{
			if (input_bit == weights[in].get_bit(out))
				accumulators[out] += 1;

		}
	}

	/*
	 * At the last clocks we threshold every accumulator.
	 */
#if MODE == MODE_MATMUL_AND_THRESHOLD

	threshold_loop: for(int out=0; out<OUTPUT_DIM; out++)
	{
		#pragma HLS PIPELINE II=1

		if (accumulators[out] > thresholds[out])
			output_stream.write(1);
		else
			output_stream.write(0);
	}

#elif MODE == MODE_MATMUL_ONLY

		output_loop: for(int p=0; p<P_OUT; p++)
		{
			#pragma HLS PIPELINE II=1
			output_stream.write(accumulators[p]);
		}

#endif

#else


	/*
	 * Second implementation: keeping OUTPUT_DIM accumulators and performing OUTPUT_DIM xnor
	 * and count costs too much. Here we keep only a subset of parallel accumulators
	 * (P_OUT | OUTPUT_DIM % P_OUT == 0) and therefore we perform less xnor and accumulate.
	 * The problem that comes in is that before switching to other output neurons, we have
	 * to finish the previous ones -> we need to cache the whole input values at run-time
	 * (not just one bit!). With this implementation we can save resources, at the cost of being slower.
	 */

	bit input_data[INPUT_DIM];

	acc_type accumulators[P_OUT];
	#pragma HLS ARRAY_PARTITION variable=accumulators complete


	for(int it=0; it < OUTPUT_DIM / P_OUT; it++)
	{
		for(int p=0; p < P_OUT; p++)
		{
#pragma HLS UNROLL
			accumulators[p] = 0;
		}

		for(int i=0; i < INPUT_DIM; i++)
		{
#pragma HLS PIPELINE II=1

			if (it == 0)
				input_data[i] = input_stream.read();


			for(int p=0; p < P_OUT; p++)
			{
				if (input_data[i] == weights[i].get_bit(it*P_OUT + p))
					accumulators[p] += 1;
			}

		}

#if MODE == MODE_MATMUL_AND_THRESHOLD

		threshold_loop: for(int p=0; p<P_OUT; p++)
		{
	#pragma HLS PIPELINE II=1
			if (accumulators[p] > thresholds[it*P_OUT + p])
				output_stream.write(1);
			else
				output_stream.write(0);
		}

#elif MODE == MODE_MATMUL_ONLY

		output_loop: for(int p=0; p<P_OUT; p++)
		{
#pragma HLS PIPELINE II=1
			output_stream.write(accumulators[p]);
		}

#endif

	}


#endif

}
