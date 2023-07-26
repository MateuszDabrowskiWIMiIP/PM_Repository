__kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C,
    const int ROW_A, const int COL_A, const int COL_B) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < COL_A; ++k) {
        sum += A[i * COL_A + k] * B[k * COL_B + j];
    }

    C[i * COL_B + j] = sum;
}
