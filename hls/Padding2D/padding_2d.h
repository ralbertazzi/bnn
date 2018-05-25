#include "ap_int.h"
#include "hls_stream.h"

#define PAD 1
#define PAD_VALUE 0
#define FMAPS 32

#define IMG_WIDTH 16
#define IMG_HEIGHT 16

typedef ap_uint<FMAPS> in_type;

void padding_2d(hls::stream<in_type> &input_stream, hls::stream<in_type> &output_stream);
