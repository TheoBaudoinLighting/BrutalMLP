// Mini-batch dense layer kernels used by training: ISPC versions of src/math/dense_batch.cpp.
//
// Same cache tiles and the same operation sequence for every value as the C++ kernels. The
// difference is the loop nest: each SIMD lane owns one value and keeps its running sum in a
// register across the whole reduction loop, so the value is loaded and stored once per tile
// instead of once per term. Built with --opt=disable-fma, every multiply and add is rounded
// separately like the C++ code, so results are bit-identical to the C++ kernels.
//
// Matrices are row-major. Row offsets are computed with 64-bit uniform integers; indices inside
// a row are 32-bit (the C++ caller checks that every dimension fits).

// z[s][o] = biases[o] + sum over i (increasing) of inputs[s][i] * weights_transposed[i][o].
export void brutal_mlp_dense_forward_batch_f32(uniform const float inputs[],
                                               uniform int64 count,
                                               uniform int32 input_size,
                                               uniform const float weights_transposed[],
                                               uniform const float biases[],
                                               uniform int32 output_size,
                                               uniform int32 tile_rows,
                                               uniform float z[]) {
    for (uniform int64 sample = 0; sample < count; ++sample) {
        uniform float* uniform row = z + sample * output_size;
        foreach (output_index = 0 ... output_size) {
            row[output_index] = biases[output_index];
        }
    }

    for (uniform int32 tile_begin = 0; tile_begin < input_size; tile_begin += tile_rows) {
        const uniform int32 tile_end = min(input_size, tile_begin + tile_rows);
        uniform int64 sample = 0;
        for (; sample + 4 <= count; sample += 4) {
            uniform const float* uniform x0 = inputs + (sample + 0) * input_size;
            uniform const float* uniform x1 = inputs + (sample + 1) * input_size;
            uniform const float* uniform x2 = inputs + (sample + 2) * input_size;
            uniform const float* uniform x3 = inputs + (sample + 3) * input_size;
            uniform float* uniform z0 = z + (sample + 0) * output_size;
            uniform float* uniform z1 = z + (sample + 1) * output_size;
            uniform float* uniform z2 = z + (sample + 2) * output_size;
            uniform float* uniform z3 = z + (sample + 3) * output_size;
            foreach (output_index = 0 ... output_size) {
                float sum0 = z0[output_index];
                float sum1 = z1[output_index];
                float sum2 = z2[output_index];
                float sum3 = z3[output_index];
                for (uniform int32 input_index = tile_begin; input_index < tile_end; ++input_index) {
                    uniform const float* uniform w = weights_transposed + (uniform int64)input_index * output_size;
                    const float weight = w[output_index];
                    sum0 += x0[input_index] * weight;
                    sum1 += x1[input_index] * weight;
                    sum2 += x2[input_index] * weight;
                    sum3 += x3[input_index] * weight;
                }
                z0[output_index] = sum0;
                z1[output_index] = sum1;
                z2[output_index] = sum2;
                z3[output_index] = sum3;
            }
        }
        for (; sample < count; ++sample) {
            uniform const float* uniform x = inputs + sample * input_size;
            uniform float* uniform row = z + sample * output_size;
            foreach (output_index = 0 ... output_size) {
                float sum = row[output_index];
                for (uniform int32 input_index = tile_begin; input_index < tile_end; ++input_index) {
                    uniform const float* uniform w = weights_transposed + (uniform int64)input_index * output_size;
                    sum += x[input_index] * w[output_index];
                }
                row[output_index] = sum;
            }
        }
    }
}

// grad_weights[o][i] += delta[s][o] * inputs[s][i] and grad_biases[o] += delta[s][o], samples in
// increasing order.
export void brutal_mlp_dense_accumulate_gradients_batch_f32(uniform const float delta[],
                                                            uniform const float inputs[],
                                                            uniform int64 count,
                                                            uniform int32 input_size,
                                                            uniform int32 output_size,
                                                            uniform int32 tile_rows,
                                                            uniform float grad_weights[],
                                                            uniform float grad_biases[]) {
    for (uniform int64 sample = 0; sample < count; ++sample) {
        uniform const float* uniform d = delta + sample * output_size;
        foreach (output_index = 0 ... output_size) {
            grad_biases[output_index] += d[output_index];
        }
    }

    // Every gradient row receives the samples in increasing order; the C++ kernel's grouping of
    // four samples does not change that order, so one sample loop is equivalent. Four rows are
    // updated together so that each loaded input value serves four rows.
    uniform int32 output_index = 0;
    for (; output_index + 4 <= output_size; output_index += 4) {
        uniform float* uniform row0 = grad_weights + (uniform int64)(output_index + 0) * input_size;
        uniform float* uniform row1 = grad_weights + (uniform int64)(output_index + 1) * input_size;
        uniform float* uniform row2 = grad_weights + (uniform int64)(output_index + 2) * input_size;
        uniform float* uniform row3 = grad_weights + (uniform int64)(output_index + 3) * input_size;
        foreach (input_index = 0 ... input_size) {
            float value0 = row0[input_index];
            float value1 = row1[input_index];
            float value2 = row2[input_index];
            float value3 = row3[input_index];
            for (uniform int64 sample = 0; sample < count; ++sample) {
                uniform const float* uniform d = delta + sample * output_size + output_index;
                const float a = inputs[sample * input_size + input_index];
                value0 += d[0] * a;
                value1 += d[1] * a;
                value2 += d[2] * a;
                value3 += d[3] * a;
            }
            row0[input_index] = value0;
            row1[input_index] = value1;
            row2[input_index] = value2;
            row3[input_index] = value3;
        }
    }
    for (; output_index < output_size; ++output_index) {
        uniform float* uniform row = grad_weights + (uniform int64)output_index * input_size;
        foreach (input_index = 0 ... input_size) {
            float value = row[input_index];
            for (uniform int64 sample = 0; sample < count; ++sample) {
                uniform const float* uniform a = inputs + sample * input_size;
                value += delta[sample * output_size + output_index] * a[input_index];
            }
            row[input_index] = value;
        }
    }
}

// input_gradient[s][i] = sum over o (increasing) of weights[o][i] * delta[s][o].
export void brutal_mlp_dense_input_gradient_batch_f32(uniform const float delta[],
                                                      uniform const float weights[],
                                                      uniform int64 count,
                                                      uniform int32 input_size,
                                                      uniform int32 output_size,
                                                      uniform int32 tile_rows,
                                                      uniform float input_gradient[]) {
    for (uniform int64 sample = 0; sample < count; ++sample) {
        uniform float* uniform g = input_gradient + sample * input_size;
        foreach (input_index = 0 ... input_size) {
            g[input_index] = 0.0f;
        }
    }

    for (uniform int32 tile_begin = 0; tile_begin < output_size; tile_begin += tile_rows) {
        const uniform int32 tile_end = min(output_size, tile_begin + tile_rows);
        uniform int64 sample = 0;
        for (; sample + 4 <= count; sample += 4) {
            uniform float* uniform g0 = input_gradient + (sample + 0) * input_size;
            uniform float* uniform g1 = input_gradient + (sample + 1) * input_size;
            uniform float* uniform g2 = input_gradient + (sample + 2) * input_size;
            uniform float* uniform g3 = input_gradient + (sample + 3) * input_size;
            uniform const float* uniform d0 = delta + (sample + 0) * output_size;
            uniform const float* uniform d1 = delta + (sample + 1) * output_size;
            uniform const float* uniform d2 = delta + (sample + 2) * output_size;
            uniform const float* uniform d3 = delta + (sample + 3) * output_size;
            foreach (input_index = 0 ... input_size) {
                float sum0 = g0[input_index];
                float sum1 = g1[input_index];
                float sum2 = g2[input_index];
                float sum3 = g3[input_index];
                for (uniform int32 output_index = tile_begin; output_index < tile_end; ++output_index) {
                    uniform const float* uniform w = weights + (uniform int64)output_index * input_size;
                    const float weight = w[input_index];
                    sum0 += weight * d0[output_index];
                    sum1 += weight * d1[output_index];
                    sum2 += weight * d2[output_index];
                    sum3 += weight * d3[output_index];
                }
                g0[input_index] = sum0;
                g1[input_index] = sum1;
                g2[input_index] = sum2;
                g3[input_index] = sum3;
            }
        }
        for (; sample < count; ++sample) {
            uniform float* uniform g = input_gradient + sample * input_size;
            uniform const float* uniform d = delta + sample * output_size;
            foreach (input_index = 0 ... input_size) {
                float sum = g[input_index];
                for (uniform int32 output_index = tile_begin; output_index < tile_end; ++output_index) {
                    uniform const float* uniform w = weights + (uniform int64)output_index * input_size;
                    sum += w[input_index] * d[output_index];
                }
                g[input_index] = sum;
            }
        }
    }
}
