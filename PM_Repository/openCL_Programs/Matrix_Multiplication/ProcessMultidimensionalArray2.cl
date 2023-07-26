__kernel void ProcessMultidimensionalArray2(const int M, const int N, const int K, global int* data, global int* data2, global int* data3)
{
	size_t id = (get_global_id(1) * get_global_size(0)) + get_global_id(0);

	const int TS = 4;

	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int globalRow = TS * get_group_id(0) + row;
	const int globalCol = TS * get_group_id(1) + col;

	__local float Asub[TS][TS];
	__local float Bsub[TS][TS];

	int acc = 0;

	const int numTiles = K / TS;

	for (int t = 0; t < numTiles; t++) {
	
		const int tiledRow = TS * t + row;
		const int tiledCol = TS * t + col;

		Asub[col][row] = data[tiledCol * M + globalRow];
		Bsub[col][row] = data2[globalCol * K + tiledRow];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < TS; k++) {
			acc += Asub[col][k] * Bsub[k][row];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	data3[globalCol * M + globalRow] = acc;
}




