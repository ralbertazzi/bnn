#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

#include "ap_int.h"
#include "hls_stream.h"

#define BIT_ACCURATE


#define HEIGHT_IMG 32
#define WIDTH_IMG 32

#define FMAPS_IN 32
#define FMAPS_OUT 32
#define K 3

#define P_IN 16
#define P_OUT 2

#define OPTIMIZED //Optimized version assumes P_IN = FMAPS_IN and P_OUT = 1

/*
 * ACC_BITS has to be equals to ceil(log2(K^2 * FMAPS_IN))
 * In the following defines we suppose K to be 3
 */

#if FMAPS_IN == 1
	#define ACC_BITS 4
#elif FMAPS_IN == 2 || FMAPS_IN == 3
	#define ACC_BITS 5
#elif FMAPS_IN >= 4 && FMAPS_IN < 8
	#define ACC_BITS 6
#elif FMAPS_IN >= 8 && FMAPS_IN < 15
	#define ACC_BITS 7
#elif FMAPS_IN >= 15 && FMAPS_IN < 29
	#define ACC_BITS 8
#elif FMAPS_IN >= 29 && FMAPS_IN < 57
	#define ACC_BITS 9
#elif FMAPS_IN >= 57 && FMAPS_IN < 114
	#define ACC_BITS 10
#elif FMAPS_IN >= 114 && FMAPS_IN < 228
	#define ACC_BITS 11
#elif FMAPS_IN >= 228 && FMAPS_IN < 456
	#define ACC_BITS 12
#elif FMAPS_IN >= 456 && FMAPS_IN < 911
	#define ACC_BITS 13
#elif FMAPS_IN >= 911
	#define ACC_BITS 14
#endif



typedef ap_uint<FMAPS_IN> in_type;
typedef ap_uint<FMAPS_OUT> out_type;
typedef ap_uint<64> u64;
typedef ap_uint<ACC_BITS> acc_type;
typedef ap_uint<1> bit;


#define K2 (K*K)

#if K == 3
	#define K_BITS 2
	#define K2_BITS 4
#elif K == 5
	#define K_BITS 3
	#define K2_BITS 5
#elif K == 7
	#define K_BITS 5
	#define K2_BITS 6
#endif



void binary_convolution(hls::stream<in_type> &input_stream, hls::stream<out_type> &output_stream,
		bit weight_interrupt, hls::stream<u64> &weight_channel);
