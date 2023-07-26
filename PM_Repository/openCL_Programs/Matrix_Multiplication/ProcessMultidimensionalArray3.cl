__kernel void ProcessMultidimensionalArray3(const int M, const int N, const int K, global int* data, global int* data2, global int* data3)
{
    const int TS = 4;
    const WPT = 1;
    const RTS = 2;
    const int row = get_local_id(0); 
    const int col = get_local_id(1); 
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col; 

    __local int Asub[TS][TS];
    __local int Bsub[TS][TS];

    int acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0;
    }

    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) {

        for (int w = 0; w < WPT; w++) {
            const int tiledRow = TS * t + row;
            const int tiledCol = TS * t + col;
            Asub[col + w * RTS][row] = data[(tiledCol + w * RTS) * M + globalRow];
            Bsub[col + w * RTS][row] = data2[(globalCol + w * RTS) * K + tiledRow];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {                       
                acc[w] += Asub[col + w * RTS][k] * Bsub[k][row];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WPT; w++) {
        data3[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}




