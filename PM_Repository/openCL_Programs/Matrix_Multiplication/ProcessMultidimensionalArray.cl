__kernel void ProcessMultidimensionalArray(const int M, const int N, const int K, global int* data, global int* data2, global int* data3)
{
	size_t id = (get_global_id(1) * get_global_size(0)) + get_global_id(0);

	const int globalRow = get_global_id(0); // Row ID of C (0..M)
	const int globalCol = get_global_id(1); // Col ID of C (0..N)

	int acc = 0.0;
	for (int k = 0; k < K; k++) {
		acc += data[globalCol * K + k] * data2[k * M + globalRow];
	}

	data3[globalCol * M + globalRow] = acc;
}




