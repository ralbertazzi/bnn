#include "classification_sender.h"


void classification_sender(hls::stream<in_type> &input_stream, uint* class_data)
{

/*
 * With the pragma HLS INTERFACE s_axilite port=return
 * Vivado HLS infers an interrupt output port that gets high
 * when the function processed everything.
 * Also, the C functions for interrupt handling are inferred
 * inside the C drivers (SDK)
 */
#pragma HLS INTERFACE s_axilite port=return bundle=BUS
#pragma HLS INTERFACE m_axi depth=10 port=class_data bundle=BUS
#pragma HLS INTERFACE axis  port=input_stream

	static u32 buffer[NUM_CLASSES];

	for(int i=0; i < NUM_CLASSES; i++)
	{
#pragma HLS PIPELINE II=1
		in_type in_val = input_stream.read();
		buffer[i] = in_val;
	}

	memcpy((uint*) class_data, (uint*) buffer, NUM_CLASSES * sizeof(uint));

}
