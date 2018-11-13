#include <assert.h>
#include <math.h>
#include "uwnet.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        float sum = 0;
        for(j = 0; j < m.cols; ++j){
            int index = i*m.cols + j;
            float x = m.data[index];
            if(a == LOGISTIC){
                // TODO
                m.data[index] = 1/(1+expf(-x));
            } else if (a == RELU){
                // TODO
                m.data[index] = (x>0)*x;
            } else if (a == LRELU){
                // TODO
                m.data[index] = (x>0) ? x : .1*x;
            } else if (a == SOFTMAX){
                // TODO
                m.data[index] = expf(x);
            }
            sum += m.data[index];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
            for(j = 0; j < m.cols; ++j){
                int index = i*m.cols + j;
                m.data[index] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            // TODO: multiply the correct element of d by the gradient
            float x = m.data[i*m.cols + j];
            if(a == LOGISTIC){
                d.data[i*m.cols + j] *= x*(1-x);
            } else if (a == RELU){
                d.data[i*m.cols + j] *= (x>0);
            } else if (a == LRELU){
                d.data[i*m.cols + j] *= (x>0) ? 1 : .1;
            }
        }
    }
}
