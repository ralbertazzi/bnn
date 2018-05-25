#include "ap_int.h"
#include "hls_stream.h"

#define IN_WIDTH 10
#define NUM_CLASSES 10

typedef ap_uint<IN_WIDTH> in_type;
typedef ap_uint<32> u32;
typedef unsigned int uint;

void classification_sender(hls::stream<in_type> &input_stream, uint* class_data);
