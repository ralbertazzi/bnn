#define INPUT_DIM 256
#define OUTPUT_DIM 256
#define P_OUT 32


#define MODE_MATMUL_ONLY 0
#define MODE_MATMUL_AND_THRESHOLD 1

#define MODE MODE_MATMUL_AND_THRESHOLD

/*
 * From the Vivado HLS manual (UG902):
 *
 * The default maximum width allowed is 1024 bits. This default may be overridden by
 * defining the macro AP_INT_MAX_W with a positive integer value less than or equal to
 * 32768 before inclusion of the ap_int.h header file.
 *
 * CAUTION! Setting the value of AP_INT_MAX_W too high may cause slow software compile and run
 * times.
 */

#if OUTPUT_DIM > 1024
	#define AP_INT_MAX_W OUTPUT_DIM
#endif

#include "hls_stream.h"
#include "ap_int.h"

#if INPUT_DIM == 1
#define ACC_BITS 1
#elif INPUT_DIM == 2 and INPUT_DIM == 3
#define ACC_BITS 2
#elif INPUT_DIM >= 4 and INPUT_DIM < 8
#define ACC_BITS 3
#elif INPUT_DIM >= 8 and INPUT_DIM < 16
#define ACC_BITS 4
#elif INPUT_DIM >= 16 and INPUT_DIM < 32
#define ACC_BITS 5
#elif INPUT_DIM >= 32 and INPUT_DIM < 64
#define ACC_BITS 6
#elif INPUT_DIM >= 64 and INPUT_DIM < 128
#define ACC_BITS 7
#elif INPUT_DIM >= 128 and INPUT_DIM < 256
#define ACC_BITS 8
#elif INPUT_DIM >= 256 and INPUT_DIM < 512
#define ACC_BITS 9
#elif INPUT_DIM >= 512 and INPUT_DIM < 1024
#define ACC_BITS 10
#elif INPUT_DIM >= 1024 and INPUT_DIM < 2048
#define ACC_BITS 11
#elif INPUT_DIM >= 2048 and INPUT_DIM < 4096
#define ACC_BITS 12
#elif INPUT_DIM >= 4096
#define ACC_BITS 13
#endif

typedef ap_uint<1> bit;
typedef ap_uint<INPUT_DIM> input_type;
typedef ap_uint<OUTPUT_DIM> output_type;
typedef ap_uint<ACC_BITS> acc_type;
typedef ap_uint<64> u64;

#if MODE == MODE_MATMUL_AND_THRESHOLD

void dense_layer(hls::stream<bit> &input_stream,
		hls::stream<bit> &output_stream,
		hls::stream<u64> &weights_channel,
		bit weights_interrupt);

#elif MODE == MODE_MATMUL_ONLY

void dense_layer(hls::stream<bit> &input_stream,
		hls::stream<acc_type> &output_stream,
		hls::stream<u64> &weights_channel,
		bit weights_interrupt);

#endif
