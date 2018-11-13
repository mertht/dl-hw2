#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // 6.1 - iterate over the input and fill in the output with max values
    // TODO: is this rite??

    for (int x = 0; x < in.rows; x++) {
        int i = x / l.stride;
        for (int y = 0; y < in.cols; y++) {
            int j = y / l.stride;
            int out_index = i * outw + j;
            float prev_val = out.data[out_index];
            float test_val = in.data[x*in.cols + y];
            if (test_val > prev_val) {
                out.data[out_index] = test_val;
            }
        }
    }

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    // TODO: boii
    for (int x = 0; x < in.rows; x++) {
        int i = x / l.stride;
        for (int y = 0; y < in.cols; y++) {
            int j = y / l.stride;
            int out_index = i * outw + j;
            float prev_val = out.data[out_index];
            float test_val = in.data[x*in.cols + y];
            if (test_val == prev_val) {
                // (x, y) was propgated fowards, so we need to pass error back
                int index = x*in.cols + y;
                assert(index >= 0);
                assert(index < delta.rows * delta.cols);
                delta.data[x*in.cols + y] = prev_delta.data[out_index];
            }
        }
    }
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

