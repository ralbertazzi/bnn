#include "ap_int.h"
#include "hls_stream.h"

#define BIT_ACCURATE

//image dim
#define HEIGHT_IMG 32
#define WIDTH_IMG  32

/*
 * Number of input feature maps from the convolution operation.
 * Although it doesn't affect the latency, because the max pooling
 * is computed in parallel, it affects the number of resources used.
 */

#define FMAPS 32

// Size of the max-pooling window (height == width == K)
#define K 2

typedef ap_uint<FMAPS> in_type;

void max_pool(hls::stream<in_type> &in_img, hls::stream<in_type> &out_img);
