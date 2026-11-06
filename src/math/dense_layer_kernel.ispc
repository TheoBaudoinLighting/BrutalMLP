export void brutal_mlp_dense_layer_f32(uniform float input[],
                                       uniform float weights[],
                                       uniform float biases[],
                                       uniform int32 input_size,
                                       uniform int32 output_size,
                                       uniform float output[]) {
    for (uniform int32 out = 0; out < output_size; ++out) {
        uniform int32 weight_offset = out * input_size;
        float lane_sum = 0.0f;

        foreach (column = 0 ... input_size) {
            lane_sum += weights[weight_offset + column] * input[column];
        }

        output[out] = biases[out] + reduce_add(lane_sum);
    }
}
