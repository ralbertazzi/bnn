#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

#define BIT_ACCURATE


#define HEIGHT_IMG 32
#define WIDTH_IMG 32

#define FAN_OUT 32
#define P_OUT 4

#define K 3

#define K2 (K*K)

typedef ap_fixed<13,5> f_type;

typedef ap_uint<FAN_OUT> out_type;
typedef ap_uint<64> u64;
typedef ap_uint<1> bit;

void fixed_point_convolution(hls::stream<f_type> &input_stream, hls::stream<out_type> &output_stream,
		bit weight_interrupt, hls::stream<u64> &weight_channel);
