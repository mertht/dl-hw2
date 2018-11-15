#include "uwnet.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>

static const float epsilon = 1e-10;

matrix mean(matrix x, int spatial)
{
    matrix m = make_matrix(1, x.cols/spatial);
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/spatial] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / spatial;
    }
    return m;
}

matrix variance(matrix x, matrix m, int spatial)
{
    matrix v = make_matrix(1, x.cols/spatial);
    // 7.1 - calculate variance

    int i, j;
    for (i = 0; i < x.rows; i++) {
        for (j = 0; j < x.cols; j++) {
            float x_ij = x.data[i*x.cols + j];
            float mean = m.data[j/spatial];

            float inner = x_ij - mean;
            v.data[j/spatial] += inner * inner;
        }
    }
    for(i = 0; i < m.cols; ++i){
        v.data[i] = v.data[i] / x.rows / spatial;
    }
    return v;
}

matrix normalize(matrix x, matrix m, matrix v, int spatial)
{
    // 7.2 - normalize array, norm = (x - mean) / sqrt(variance + eps)
    matrix norm = make_matrix(x.rows, x.cols);

    int i, j;
    for (i = 0; i < x.rows; i++) {
        for (j = 0; j < x.cols; j++) {
            float xij = x.data[i*x.cols + j];
            float mean = m.data[j/spatial];
            float var = v.data[j/spatial];

            norm.data[i*x.cols + j] = (xij - mean) / sqrt(var + epsilon);
        }
    }
    return norm;
}

matrix batch_normalize_forward(layer l, matrix x)
{
    
    float s = .1;
    int spatial = x.cols / l.rolling_mean.cols;
    if (x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
    }
    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix x_norm = normalize(x, m, v, spatial);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);

    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    free_matrix(l.x[0]);
    l.x[0] = x;

    return x_norm;
}


matrix delta_mean(matrix d, matrix variance, int spatial)
{
    matrix dm = make_matrix(1, variance.cols);
    // 7.3 - calculate dL/dmean

    int i, j;
    for (i = 0; i < d.rows; i++) {
        for (j = 0; j < d.cols; j++) {
            float dxhat_ij = d.data[i*d.cols + j];
            float var = variance.data[j/spatial];
            float term2 = -1 / (var + epsilon);
            dm.data[j/spatial] += dxhat_ij * term2;
        }
    }
    
    return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial)
{
    matrix dv = make_matrix(1, variance.cols);
    // 7.4 - calculate dL/dvariance

    int i, j;
    for (i = 0; i < d.rows; i++) {
        for (j = 0; j < d.cols; j++) {
            float x_ij = x.data[i*x.cols + j];
            float dxhat_ij = d.data[i*d.cols + j];
            float partB = x_ij - mean.data[j/spatial];

            float inner = variance.data[j/spatial] + epsilon;
            float partC = -0.5 * powf(inner, -1.5);

            dv.data[j/spatial] += dxhat_ij * partB * partC;
        }
    }
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial)
{
    matrix dx = make_matrix(d.rows, d.cols);
    // 7.5 - calculate dL/dx

    int m = d.rows * spatial;

    int i, j;
    for (i = 0; i < d.rows; i++) {
        for (j = 0; j < d.cols; j++) {

            float x_ij = x.data[i*x.cols + j];
            float mean_j = mean.data[j/spatial];
            float var_j = variance.data[j/spatial];

            float dxhat_ij = d.data[i*d.cols + j];
            float dm_j = dm.data[j/spatial];
            float dv_j = dv.data[j/spatial];

            float part1B = 1 / sqrt(var_j + epsilon);
            float term1 = dxhat_ij * part1B;

            float part2B = 2 * (x_ij - mean_j) / m;
            float term2 = dv_j * part2B;

            float term3 = dm_j / m;

            dx.data[i*x.cols + j] = term1 + term2 + term3;
        }
    }
    return dx;
}

matrix batch_normalize_backward(layer l, matrix d)
{
    int spatial = d.cols / l.rolling_mean.cols;
    matrix x = l.x[0];

    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix dm = delta_mean(d, v, spatial);
    matrix dv = delta_variance(d, x, m, v, spatial);
    matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}
