#define PADDING 0
#define NR_ITER 1000
#define BLOCK_SIZE 16
#define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
#define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

__kernel void mat_transp_2_kernel(
    __global float* a,
    __global float* a_t,
    unsigned int a_width,
    unsigned int a_height
)
{

    int read_idx = get_global_id(0) + get_global_id(1) * a_width;
    int write_idx = get_global_id(1) + get_global_id(0) * a_height;

    a_t[write_idx] = a[read_idx];

    //__local float a_local [(BLOCK_SIZE+PADDING)*BLOCK_SIZE];

    //int base_idx_a = 

    //int local_shift = 

    //int glob_idx_a = base_idx_a + local_shift;

    //int base_idx_a_t =

    //int glob_idx_a_t = base_idx_a_t + local_shift;

    //a_local [ *(BLOCK_SIZE+PADDING) + ] = a [glob_idx_a];
    //barrier (CLK_LOCAL_MEM_FENCE);

    // miejsce na dodatkowy kod...

    //a_t[glob_idx_a_t] = a_local[ *(BLOCK_SIZE+PADDING) + ];

}


/* int i; */
/* for(i=0; i<NR_ITER; i++){ */

/*   a_local [ get_local_id (1)*(BLOCK_SIZE+PADDING) + get_local_id (0)] = */
/*     a_local [ get_local_id (1)*(BLOCK_SIZE+PADDING) + get_local_id (0)] * */
/*     a_local[ get_local_id(0)*(BLOCK_SIZE+PADDING) + get_local_id(1)]; */

/* } */

/* barrier (CLK_LOCAL_MEM_FENCE); */
