// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include<stdlib.h>
#include<stdio.h>
#include <math.h>
#include <iostream>

#include <CL/cl.h>

#include <chrono>

#include"OpenCL_util.h"

using namespace std::chrono;

using namespace std;

#define time_measurments

#ifdef time_measurments
static double t_begin, t_end, t_total;
#endif



#define BLOCK_SIZE 16
#define NR_GROUPS 16
#define MULT 16
#define WYMIAR 1024
#define ROZMIAR (WYMIAR*WYMIAR)
// Matrices are stored in row-major order: 
// M(row, col) = M( row * WYMIAR + col ) 


#define WORK_GROUP_SIZE 256

#define time_measurments

// for single or double precision calculations
// do not forget to choose the proper kernel!
//#define SCALAR double
#define SCALAR float

void displayInfo();

void DisplayPlatformInfo(
    cl_platform_id id,
    cl_platform_info name,
    char* str)
{
    cl_int retval;
    size_t paramValueSize;

    retval = clGetPlatformInfo(
        id,
        name,
        0,
        NULL,
        &paramValueSize);
    if (retval != CL_SUCCESS) {
        printf("Failed to find OpenCL platform %s.\n", str);
        return;
    }

    char* info = (char*)malloc(sizeof(char) * paramValueSize);
    retval = clGetPlatformInfo(
        id,
        name,
        paramValueSize,
        info,
        NULL);
    if (retval != CL_SUCCESS) {
        printf("Failed to find OpenCL platform %s.\n", str);
        return;
    }

    printf("\t%s:\t%s\n", str, info);
    free(info);
}

void DisplayDeviceInfo(
    cl_device_id id,
    cl_device_info name,
    char* str)
{
    cl_int retval;
    size_t paramValueSize;

    retval = clGetDeviceInfo(
        id,
        name,
        0,
        NULL,
        &paramValueSize);
    if (retval != CL_SUCCESS) {
        printf("Failed to find OpenCL device info %s.\n", str);
        return;
    }

    char* info = (char*)malloc(sizeof(char) * paramValueSize);
    retval = clGetDeviceInfo(
        id,
        name,
        paramValueSize,
        info,
        NULL);

    if (retval != CL_SUCCESS) {
        printf("Failed to find OpenCL device info %s.\n", str);
        return;
    }

    printf("\t\t%s:\t%s\n", str, info);
    free(info);
};


int utr_ocl_create_contexts(
    int Chosen_platform_id,
    int Monitor,
    utt_ocl_struct& utv_ocl_struct
)
{
    cl_int retval;
    cl_uint numPlatforms;
    cl_platform_id* platformIds;
    cl_context context = NULL;
    cl_uint i, j, k;

    // First, query the total number of platforms
    retval = clGetPlatformIDs(0, (cl_platform_id*)NULL, &numPlatforms);

    // allocate memory for local platform structures
    utv_ocl_struct.number_of_platforms = numPlatforms;
    utv_ocl_struct.list_of_platforms =
        (utt_ocl_platform_struct*)malloc(sizeof(utt_ocl_platform_struct)
            * numPlatforms);

    // Next, allocate memory for the installed platforms, and qeury 
    // to get the list.
    platformIds = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    retval = clGetPlatformIDs(numPlatforms, platformIds, NULL);

    if (Monitor >= UTC_BASIC_INFO) {
        printf("\nNumber of OpenCL platforms: \t%d\n", numPlatforms);
    }

    // Iterate through the list of platforms displaying associated information
    for (i = 0; i < numPlatforms; i++) {

        if (Monitor > UTC_BASIC_INFO) {
            printf("\n");
            printf("Platform %d:\n", i);
        }

        utv_ocl_struct.list_of_platforms[i].id = platformIds[i];
        //clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, size_of_name???, 
        //		      utv_ocl_struct.list_of_platforms[i].name, (size_t *) NULL);

        if (Monitor > UTC_BASIC_INFO) {

            // First we display information associated with the platform
            DisplayPlatformInfo(
                platformIds[i],
                CL_PLATFORM_NAME,
                (char*)"CL_PLATFORM_NAME");
            DisplayPlatformInfo(
                platformIds[i],
                CL_PLATFORM_PROFILE,
                (char*)"CL_PLATFORM_PROFILE");
            DisplayPlatformInfo(
                platformIds[i],
                CL_PLATFORM_VERSION,
                (char*)"CL_PLATFORM_VERSION");
            DisplayPlatformInfo(
                platformIds[i],
                CL_PLATFORM_VENDOR,
                (char*)"CL_PLATFORM_VENDOR");
        }

        // Now query the set of devices associated with the platform
        cl_uint numDevices;
        retval = clGetDeviceIDs(
            platformIds[i],
            CL_DEVICE_TYPE_ALL,
            0,
            NULL,
            &numDevices);


        utv_ocl_struct.list_of_platforms[i].number_of_devices = numDevices;
        utv_ocl_struct.list_of_platforms[i].list_of_devices =
            (utt_ocl_device_struct*)malloc(sizeof(utt_ocl_device_struct)
                * numDevices);

        cl_device_id* devices =
            (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);

        retval = clGetDeviceIDs(
            platformIds[i],
            CL_DEVICE_TYPE_ALL,
            numDevices,
            devices,
            NULL);

        if (Monitor >= UTC_BASIC_INFO) {
            printf("\n\tNumber of devices: \t%d\n", numDevices);
        }
        // Iterate through each device, displaying associated information
        for (j = 0; j < numDevices; j++)
        {

            if (Monitor > UTC_BASIC_INFO) {
                printf("\tDevice %d:\n", j);
            }
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].id =
                devices[j];
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type),
                &utv_ocl_struct.list_of_platforms[i].list_of_devices[j].type, NULL);

            if (utv_ocl_struct.list_of_platforms[i].list_of_devices[j].type == CL_DEVICE_TYPE_CPU) {
                utv_ocl_struct.list_of_platforms[i].list_of_devices[j].utc_type = UTC_OCL_DEVICE_CPU;
            }
            if (utv_ocl_struct.list_of_platforms[i].list_of_devices[j].type == CL_DEVICE_TYPE_GPU) {
                utv_ocl_struct.list_of_platforms[i].list_of_devices[j].utc_type = UTC_OCL_DEVICE_GPU;
            }
            if (utv_ocl_struct.list_of_platforms[i].list_of_devices[j].type == CL_DEVICE_TYPE_ACCELERATOR) {
                utv_ocl_struct.list_of_platforms[i].list_of_devices[j].utc_type = UTC_OCL_DEVICE_ACCELERATOR;
            }

            cl_ulong mem_size_ulong = 0;
            int err_num = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                sizeof(cl_ulong), &mem_size_ulong, NULL);
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].global_mem_bytes =
                (double)mem_size_ulong;

            err_num = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                sizeof(cl_ulong), &mem_size_ulong, NULL);
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].global_max_alloc =
                (double)mem_size_ulong;

            err_num = clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE,
                sizeof(cl_ulong), &mem_size_ulong, NULL);
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].local_mem_bytes =
                (double)mem_size_ulong;

            err_num = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                sizeof(cl_ulong), &mem_size_ulong, NULL);
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].cache_bytes =
                (double)mem_size_ulong;

            err_num = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                sizeof(cl_ulong), &mem_size_ulong, NULL);
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].constant_mem_bytes =
                (double)mem_size_ulong;

            cl_uint cache_line_size = 0;
            err_num = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                sizeof(cl_uint), &cache_line_size, NULL);
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].cache_line_bytes =
                (int)cache_line_size;

            cl_uint max_num_comp_units = 0;
            err_num = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(cl_uint), &max_num_comp_units, NULL);
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].max_num_comp_units =
                (int)max_num_comp_units;

            size_t max_work_group_size = 0;
            err_num = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                sizeof(size_t), &max_work_group_size, NULL);
            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].max_work_group_size = (int)max_work_group_size;

            // possible further inquires:
            //CL_DEVICE_MAX_WORK_GROUP_SIZE, 
            //CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_DEVICE_MAX_WORK_ITEM_SIZES
            //CL_DEVICE_MAX_CONSTANT_ARGS
            //CL_DEVICE_MAX_PARAMETER_SIZE
            //CL_DEVICE_PREFERRED_VECTOR_WIDTH_ - char, int, float, double etc.
            //CL_DEVICE_MEM_BASE_ADDR_ALIGN, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE

            utv_ocl_struct.list_of_platforms[i].list_of_devices[j].command_queue = 0;

            for (k = 0; k < UTC_OCL_MAX_NUM_KERNELS; k++) {
                utv_ocl_struct.list_of_platforms[i].list_of_devices[j].program[k] = 0;
                utv_ocl_struct.list_of_platforms[i].list_of_devices[j].kernel[k] = 0;
            }

            //clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name?), 
            //&utv_ocl_struct.list_of_platforms[i].list_of_devices[j].name, NULL);

            if (Monitor > UTC_BASIC_INFO) {

                DisplayDeviceInfo(
                    devices[j],
                    CL_DEVICE_NAME,
                    (char*)"CL_DEVICE_NAME");

                DisplayDeviceInfo(
                    devices[j],
                    CL_DEVICE_VENDOR,
                    (char*)"CL_DEVICE_VENDOR");

                DisplayDeviceInfo(
                    devices[j],
                    CL_DEVICE_VERSION,
                    (char*)"CL_DEVICE_VERSION");
                printf("\t\tdevice global memory size (MB) = %lf\n",
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].global_mem_bytes / 1024 / 1024);
                printf("\t\tdevice global max alloc size (MB) = %lf\n",
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].global_max_alloc / 1024 / 1024);
                printf("\t\tdevice local memory size (kB) = %lf\n",
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].local_mem_bytes / 1024);
                printf("\t\tdevice constant memory size (kB) = %lf\n",
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].constant_mem_bytes / 1024);
                printf("\t\tdevice cache memory size (kB) = %lf\n",
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].cache_bytes / 1024);
                printf("\t\tdevice cache line size (B) = %d\n",
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].cache_line_bytes);
                printf("\t\tdevice maximal number of compute units = %d\n",
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].max_num_comp_units);
                printf("\t\tdevice maximal number of work units in work group = %d\n",
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].max_work_group_size);

                printf("\n");
            }
        }

        free(devices);

        // Next, create OpenCL contexts on platforms
        cl_context_properties contextProperties[] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)platformIds[i],
          0
        };

        if (Chosen_platform_id == UTC_OCL_ALL_PLATFORMS || Chosen_platform_id == i) {

            if (Monitor > UTC_BASIC_INFO) {
                printf("\tCreating CPU context (index=0) on platform %d\n", i);
            }

            utv_ocl_struct.list_of_platforms[i].list_of_contexts[0] =
                clCreateContextFromType(contextProperties,
                    CL_DEVICE_TYPE_CPU, NULL, NULL, &retval);

            if (Monitor >= UTC_BASIC_INFO && retval != CL_SUCCESS) {
                printf("\tCould not create CPU context on platform %d\n", i);
            }

            if (Monitor > UTC_BASIC_INFO) {
                printf("\tCreating GPU context (index=1) on platform %d\n", i);
            }

            utv_ocl_struct.list_of_platforms[i].list_of_contexts[1] =
                clCreateContextFromType(contextProperties,
                    CL_DEVICE_TYPE_GPU, NULL, NULL, &retval);

            if (Monitor >= UTC_BASIC_INFO && retval != CL_SUCCESS) {
                printf("\tCould not create GPU context on platform %d\n", i);
            }

            if (Monitor > UTC_BASIC_INFO) {
                printf("\tCreating ACCELERATOR context (index=2) on platform %d\n", i);
            }

            utv_ocl_struct.list_of_platforms[i].list_of_contexts[2] =
                clCreateContextFromType(contextProperties,
                    CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, &retval);
            if (Monitor >= UTC_BASIC_INFO && retval != CL_SUCCESS) {
                printf("\tCould not create ACCELERATOR context on platform %d\n", i);
            }

        }
    }

    free(platformIds);
    return numPlatforms;
}

int utr_ocl_create_command_queues(
    int Chosen_platform_index,
    int Chosen_device_type,
    int Monitor,
    utt_ocl_struct utv_ocl_struct
)
{

    // in a loop over all platforms
    int platform_index;
    for (platform_index = 0;
        platform_index < utv_ocl_struct.number_of_platforms;
        platform_index++) {

        // shortcut for global platform structure
        utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[platform_index];

        // if creating contexts for all platforms or just this one 
        if (Chosen_platform_index == UTC_OCL_ALL_PLATFORMS ||
            Chosen_platform_index == platform_index) {

            // in a loop over all devices
            int idev;
            for (idev = 0;
                idev < platform_struct.number_of_devices;
                idev++) {

                // variable for storing device_id
                cl_device_id device = 0;

                // select context for the device (CPU context for CPU device, etc.)
                // (contexts are already created!,
                // icon is just the index in the platform structure)	
                int icon;

                // check whether this is a CPU device - then context is no 0
                if (platform_struct.list_of_devices[idev].type ==
                    CL_DEVICE_TYPE_CPU) {

                    if (Chosen_device_type == UTC_OCL_ALL_DEVICES ||
                        Chosen_device_type == UTC_OCL_DEVICE_CPU) {

                        device = platform_struct.list_of_devices[idev].id;
                        platform_struct.list_of_devices[idev].utc_type = UTC_OCL_DEVICE_CPU;
                        icon = 0;

                    }
                    else {

                        device = NULL;

                    }

                }
                // check whether this is a GPU device - then context is no 1
                else if (platform_struct.list_of_devices[idev].type ==
                    CL_DEVICE_TYPE_GPU) {

                    if (Chosen_device_type == UTC_OCL_ALL_DEVICES ||
                        Chosen_device_type == UTC_OCL_DEVICE_GPU) {

                        device = platform_struct.list_of_devices[idev].id;
                        platform_struct.list_of_devices[idev].utc_type = UTC_OCL_DEVICE_GPU;
                        icon = 1;

                    }
                    else {

                        device = NULL;

                    }

                }
                // check whether this is an ACCELERATOR device - then context is no 2
                else if (platform_struct.list_of_devices[idev].type ==
                    CL_DEVICE_TYPE_ACCELERATOR) {

                    if (Chosen_device_type == UTC_OCL_ALL_DEVICES ||
                        Chosen_device_type == UTC_OCL_DEVICE_ACCELERATOR) {

                        device = platform_struct.list_of_devices[idev].id;
                        platform_struct.list_of_devices[idev].utc_type = UTC_OCL_DEVICE_ACCELERATOR;
                        icon = 2;

                    }
                    else {

                        device = NULL;

                    }

                }

                if (device != NULL) {

                    // choose OpenCL context selected for a device
                    cl_context context = platform_struct.list_of_contexts[icon];
                    platform_struct.list_of_devices[idev].context_index = icon;

                    // if context exist
                    if (context != NULL) {

                        if (Monitor > UTC_BASIC_INFO) {
                            if (platform_struct.list_of_devices[idev].utc_type == UTC_OCL_DEVICE_CPU) {
                                printf("\nCreating command queue for CPU context %d, device %d, platform %d\n",
                                    icon, idev, platform_index);
                            }
                            if (platform_struct.list_of_devices[idev].utc_type == UTC_OCL_DEVICE_GPU) {
                                printf("\nCreating command queue for GPU context %d, device %d, platform %d\n",
                                    icon, idev, platform_index);
                            }
                            if (platform_struct.list_of_devices[idev].utc_type == UTC_OCL_DEVICE_ACCELERATOR) {
                                printf("\nCreating command queue for ACCELERATOR context %d, device %d, platform %d\n",
                                    icon, idev, platform_index);
                            }
                        }

                        // Create a command-queue on the device for the context
                        cl_command_queue_properties prop = 0;
                        prop |= CL_QUEUE_PROFILING_ENABLE;
                        platform_struct.list_of_devices[idev].command_queue =
                            clCreateCommandQueue(context, device, prop, NULL);
                        if (platform_struct.list_of_devices[idev].command_queue == NULL)
                        {
                            printf("Failed to create command queue for context %d, device %d, platform %d\n",
                                icon, idev, platform_index);
                            exit(-1);
                        }

                    } // end if context exist for a given device

                } // end if device is of specified type

            } // end loop over devices

        } // end if platform is of specified type

    } // end loop over platforms

    return(0);
}

char* utr_ocl_readSource(const char* kernelPath) {

    cl_int status;
    FILE* fp;
    char* source;
    long int size;

    fp = fopen(kernelPath, "rb");
    if (!fp) {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    status = fseek(fp, 0, SEEK_END);
    if (status != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if (size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char*)malloc(size + 1);

    int i;
    for (i = 0; i < size + 1; i++) {
        source[i] = '\0';
    }

    if (source == NULL) {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }

    fread(source, 1, size, fp);
    source[size] = '\0';

    return source;
}

int utr_ocl_create_kernel(
    int Platform_index,
    int Device_index,
    int Kernel_index,
    char* Kernel_name,
    const char* Kernel_file,
    int Monitor,
    utt_ocl_struct utv_ocl_struct
)
{

    cl_int retval;

    // choose the platform
    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];

    // check the device !!!!!!!!!!!!!!!!! (or at least its index)
    if (Device_index < 0) {
        printf("Wrong device_index %d passed to utr_ocl_create_kernel! Exiting.\n",
            Device_index);
        exit(-1);
    }

    cl_device_id device = platform_struct.list_of_devices[Device_index].id;
    cl_context context = platform_struct.list_of_contexts[
        platform_struct.list_of_devices[Device_index].context_index
    ];

    if (Monitor > UTC_BASIC_INFO) {
        printf("Program file is: %s\n", Kernel_file);
    }

    // read source file into data structure
    const char* source = utr_ocl_readSource(Kernel_file);



    cl_program program = clCreateProgramWithSource(context, 1,
        &source,
        NULL, NULL);
    if (program == NULL)
    {
        printf("Failed to create CL program from source.\n");
        exit(-1);
    }

    // TO GET INFO FROM NVIDIA COMPILER
    //retval = clBuildProgram(program, 0, NULL, "-cl-nv-verbose", NULL, NULL); 
    retval = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    char* buildLog; size_t size_of_buildLog;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
        0, NULL, &size_of_buildLog);
    buildLog = (char*)malloc(size_of_buildLog + 1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
        size_of_buildLog, buildLog, NULL);
    buildLog[size_of_buildLog] = '\0';
    printf("Kernel buildLog: %s\n", buildLog);
    if (retval != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), buildLog, NULL);

        printf("Error in kernel\n");
        clReleaseProgram(program);
        exit(-1);
        //return NULL;
    }


    // Create OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, Kernel_name, NULL);
    if (kernel == NULL)
    {
        printf("Failed to create kernel.\n");
        exit(-1);
        //  return 1;
    }

    if (Monitor > UTC_BASIC_INFO) {
        printf("Created kernel for platform %d, device %d, kernel index %d\n",
            Platform_index, Device_index, Kernel_index);
    }

    platform_struct.list_of_devices[Device_index].program[Kernel_index] = program;
    platform_struct.list_of_devices[Device_index].kernel[Kernel_index] = kernel;

    return(0);
}


int utr_ocl_create_kernel_dev_type(
    int Platform_index,
    int Device_type,
    int Kernel_index,
    char* Kernel_name,
    const char* Kernel_file,
    int Monitor,
    utt_ocl_struct utv_ocl_struct
)
{

    cl_int retval;

    // choose the platform
    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];

    // choose the device
    int idev; int device_index;
    for (idev = 0;
        idev < platform_struct.number_of_devices;
        idev++) {

        if (platform_struct.list_of_devices[idev].utc_type == Device_type) {

            if (Monitor > UTC_BASIC_INFO) {
                if (Device_type == UTC_OCL_DEVICE_CPU) {
                    printf("\nCreating kernel %d for platform %d: selected device %d for type %d (CPU)\n",
                        Kernel_index, Platform_index, idev, Device_type);
                }
                if (Device_type == UTC_OCL_DEVICE_GPU) {
                    printf("\nCreating kernel %d for platform %d: selected device %d for type %d (GPU)\n",
                        Kernel_index, Platform_index, idev, Device_type);
                }
                if (Device_type == UTC_OCL_DEVICE_ACCELERATOR) {
                    printf("\nCreating kernel %d for platform %d: selected device %d for type %d (ACCELERATOR)\n",
                        Kernel_index, Platform_index, idev, Device_type);
                }
            }

            device_index = idev;
            break;

        }

    }

    utr_ocl_create_kernel(Platform_index, device_index, Kernel_index,
        Kernel_name, Kernel_file, Monitor, utv_ocl_struct);

    return(0);
}


int utr_ocl_GPU_context_exists(
    int Platform_index,
    utt_ocl_struct utv_ocl_struct
)
{
    // choose the platform
    //printf((char *)utv_ocl_struct.list_of_platforms[Platform_index]);

    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
    if (platform_struct.list_of_contexts[1] == NULL) return(0);
    return(1);
}

void create_kernels(utt_ocl_struct utv_ocl_struct)
{
    // for all operations indicate explicit info messages
    int monitor = UTC_BASIC_INFO + 1;

    int kernel_index;

    // calculations are performed for the selected platform
    int platform_index = utv_ocl_struct.current_platform_index;

    if (utr_ocl_GPU_context_exists(platform_index, utv_ocl_struct)) {



        // create the first kernel for GPU
        kernel_index = 0;
        utr_ocl_create_kernel_dev_type(platform_index, UTC_OCL_DEVICE_GPU, kernel_index,
            // kernel name:         , file:
            (char*)"mat_transp_1_kernel", "mat_transp_1.cl", monitor, utv_ocl_struct);

        // create the second kernel for GPU
        kernel_index = 1;
        utr_ocl_create_kernel_dev_type(platform_index, UTC_OCL_DEVICE_GPU, kernel_index,
            // kernel name:         , file:
            (char*)"mat_transp_2_kernel", "mat_transp_2.cl", monitor, utv_ocl_struct);

        // create the third kernel for GPU
        kernel_index = 2;
        utr_ocl_create_kernel_dev_type(platform_index, UTC_OCL_DEVICE_GPU, kernel_index,
            // kernel name:         , file:
            (char*)"mat_transp_3_kernel", "mat_transp_3.cl", monitor, utv_ocl_struct);

    }
}

int verify_result(
    float* result,
    float* result_compare
)
{
    // Verify the result
    int result_OK = 1;
    int i, j;
    for (i = 0; i < WYMIAR; i++) {
        for (j = 0; j < WYMIAR; j++) {
            if (fabs(result[i * WYMIAR + j] - result_compare[i + j * WYMIAR]) > 1.e-6) {
                result_OK = 0;
                break;
            }
        }
    }
    printf("\t\t6. verifying results: ");
    if (result_OK) {
        printf("Output is correct\n");
    }
    else {
        printf("Output is incorrect\n");
        getchar();
        getchar();
        /* for(i = 0; i < WYMIAR; i++) { */
        /*   for(j = 0; j < WYMIAR; j++) { */
        /*   //for(i = 0; i < 10; i++) { */
        /* 	if(fabs(result[i*WYMIAR+j] - result_compare[i+j*WYMIAR])>1.e-9) { */
        /* 	  printf("%d %d %16.8f %16.8f\n",  */
        /* 	  	 i, j, result[i*WYMIAR+j], result_compare[i+WYMIAR*j]); */
        /* 	  getchar(); */
        /* 	} */
        /*   } */
        /* } */
        /* exit(0); */
    }
    /* for(i = 0; i < length; i++) { */
    /*   printf("%16.8f %16.8f\n", result[i], result_compare[i]); */
    /* } */

    return(result_OK);
}

int OpenCL_Hello_host_2(
    int kernel_index,
    SCALAR* A,
    SCALAR* B,
    SCALAR* C,
    int N,
    int work_group_size,
    const cl_context context,
    const cl_kernel OpenCL_Hello_kernel,
    const cl_command_queue queue
)
{

    int retval;

    // Allocate A in device memory 
    size_t size_bytes = N * sizeof(SCALAR);
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,
        size_bytes, NULL, NULL);

    // Write A to device memory 
    clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, size_bytes, A, 0, 0, 0);

    // Allocate B in device memory 
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY,
        size_bytes, NULL, NULL);

    // Write B to device memory 
    clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, size_bytes, B, 0, 0, 0);

    // Allocate C in device memory 
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_bytes, NULL, NULL);


    // Invoke kernel 
    retval = clSetKernelArg(OpenCL_Hello_kernel, 0, sizeof(int), (void*)&N);
    retval |= clSetKernelArg(OpenCL_Hello_kernel, 1, sizeof(cl_mem), (void*)&d_A);
    retval |= clSetKernelArg(OpenCL_Hello_kernel, 2, sizeof(cl_mem), (void*)&d_B);
    retval |= clSetKernelArg(OpenCL_Hello_kernel, 3, sizeof(cl_mem), (void*)&d_C);
    if (retval != CL_SUCCESS) {
        printf("Failed to Set the kernel arguments.\n");
        exit(-1);
    }

    size_t globalWorkSize[3] = { N, 0, 0 };
    size_t localWorkSize[3] = { work_group_size, 0, 0 };
    cl_uint work_dim = 1;

    A[0] = 2;
    B[0] = 2;

    // wait for previous events to finish
    clFinish(queue);
    // Enqueue a kernel run call
    cl_event ndrEvt;
    clEnqueueNDRangeKernel(queue, OpenCL_Hello_kernel, work_dim, 0,
        globalWorkSize, localWorkSize, 0, 0, &ndrEvt);
    clWaitForEvents(1, &ndrEvt);
    clFinish(queue);
    // Calculate performance 
    cl_ulong startTime;
    cl_ulong endTime;

    // Get kernel profiling info 
    clGetEventProfilingInfo(ndrEvt,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &startTime,
        0);
    clGetEventProfilingInfo(ndrEvt,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &endTime,
        0);

    // Read B from device memory 
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size_bytes, C, 0, 0, 0);

    // Free device memory 
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);


    return(0);

}

int utr_ocl_device_type(
    int Platform_index,
    int Device_index,
    utt_ocl_struct utv_ocl_struct
)
{
    // choose the platform
    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
    return(platform_struct.list_of_devices[Device_index].utc_type);
}

cl_context utr_ocl_select_context(
    int Platform_index,
    int Device_index,
    utt_ocl_struct utv_ocl_struct
)
{
    // choose the platform
    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
    int context_index = platform_struct.list_of_devices[Device_index].context_index;
    return(platform_struct.list_of_contexts[context_index]);
}

cl_command_queue utr_ocl_select_command_queue(
    int Platform_index,
    int Device_index,
    utt_ocl_struct utv_ocl_struct
)
{
    // choose the platform
    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
    return(platform_struct.list_of_devices[Device_index].command_queue);
}


cl_kernel utr_ocl_select_kernel(
    int Platform_index,
    int Device_index,
    int Kernel_index,
    utt_ocl_struct utv_ocl_struct
)
{
    // choose the platform
    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
    return(platform_struct.list_of_devices[Device_index].kernel[Kernel_index]);
}


// Matrix transposition - Host code 
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE 
int mat_transp_host(
    int kernel_index,
    float* A,
    float* B,
    int N,
    const cl_context context,
    const cl_kernel mat_transp_kernel,
    const cl_command_queue queue
)
{

    // Load A to device memory 
    size_t size_bytes = N * N * sizeof(float);
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,
        size_bytes, NULL, NULL);

    // Write A to device memory 
    clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, size_bytes, A, 0, 0, 0);

    // Allocate B in device memory 
    cl_mem d_B = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_bytes, NULL, NULL);


    // Invoke kernel 
    clSetKernelArg(mat_transp_kernel, 0, sizeof(cl_mem), (void*)&d_A);
    clSetKernelArg(mat_transp_kernel, 1, sizeof(cl_mem), (void*)&d_B);
    clSetKernelArg(mat_transp_kernel, 2, sizeof(int), (void*)&N);
    clSetKernelArg(mat_transp_kernel, 3, sizeof(int), (void*)&N);

    size_t localWorkSize[3];
    size_t globalWorkSize[3];
    cl_uint work_dim;

    if (kernel_index == 2) {

        work_dim = 2;
        localWorkSize[0] = 32;
        globalWorkSize[0] = N;
        localWorkSize[1] = 32 / 4;
        globalWorkSize[1] = N / 4;
        localWorkSize[2] = 0;
        globalWorkSize[2] = 0;


    }
    else {

        work_dim = 2;
        localWorkSize[0] = BLOCK_SIZE;
        globalWorkSize[0] = N;
        localWorkSize[1] = BLOCK_SIZE;
        globalWorkSize[1] = N;
        localWorkSize[2] = 0;
        globalWorkSize[2] = 0;

    }

    clFinish(queue);
    // Enqueue a kernel run call
    cl_event ndrEvt;
    clEnqueueNDRangeKernel(queue, mat_transp_kernel, work_dim, 0,
        globalWorkSize, localWorkSize, 0, 0, &ndrEvt);
    clWaitForEvents(1, &ndrEvt);
    clFinish(queue);
    // Calculate performance 
    cl_ulong startTime;
    cl_ulong endTime;

    // Get kernel profiling info 
    clGetEventProfilingInfo(ndrEvt,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &startTime,
        0);
    clGetEventProfilingInfo(ndrEvt,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &endTime,
        0);
    double time = (double)endTime - (double)startTime;



    // Read B from device memory 
    clEnqueueReadBuffer(queue, d_B, CL_TRUE, 0, size_bytes, B, 0, 0, 0);

    // Free device memory 
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);

    return(0);

}

void execute_kernels(utt_ocl_struct utv_ocl_struct)
{
    // for all operations indicate explicit info messages
    int monitor = UTC_BASIC_INFO + 1;

    // calculations are performed for the first platform (platform_index == 0)
    int platform_index = utv_ocl_struct.current_platform_index;
    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[platform_index];

    int kernel_index;
    int i, j, n;

    double nr_access;
    double t1, t2;

    if (monitor > UTC_BASIC_INFO) {
        printf("\n------------Starting execution phase----------------\n");
    }

    // create matrices
    n = WYMIAR;
    float* A = (float*)malloc(ROZMIAR * sizeof(float));
    float* B = (float*)malloc(ROZMIAR * sizeof(float));
    float* C = (float*)malloc(ROZMIAR * sizeof(float));

    for (i = 0; i < ROZMIAR; i++) A[i] = 1.0 * i / 1000000.0;

    nr_access = 2.0 * ROZMIAR; // read + write

    printf("mat_transp: nr_access %lf\n", nr_access);

    // get hardware characteristics to select good matrix shape
    // the set of device characteristics stored in data structure
    int device_index = 0;
    utt_ocl_device_struct device_struct =
        utv_ocl_struct.list_of_platforms[platform_index].list_of_devices[device_index];
    double global_mem_bytes = device_struct.global_mem_bytes;
    double global_max_alloc = device_struct.global_max_alloc;
    double local_mem_bytes = device_struct.local_mem_bytes;
    double constant_mem_bytes = device_struct.constant_mem_bytes;
    int max_num_comp_units = device_struct.max_num_comp_units;
    int max_work_group_size = device_struct.max_work_group_size;

    // in a loop over devices (or for a selected device)
    int idev = 0;
    for (idev = 0; idev < platform_struct.number_of_devices; idev++) {

        // int device_type = .....
        // choose device_index
        // int device_index = utr_ocl_select_device(platform_index, device_type);
        int device_index = idev;
        int device_type = utr_ocl_device_type(platform_index, device_index, utv_ocl_struct);

        if (device_index > 0 && device_type == utr_ocl_device_type(platform_index, device_index - 1, utv_ocl_struct)) break;
        //if(device_type == UTC_OCL_DEVICE_CPU) break;
        // choose the context
        cl_context context = utr_ocl_select_context(platform_index, device_index, utv_ocl_struct);

        // choose the command queue
        cl_command_queue command_queue =
            utr_ocl_select_command_queue(platform_index, device_index, utv_ocl_struct);

        if (monitor > UTC_BASIC_INFO) {
            printf("\nExecution: \t0. restoring context and command queue for platform %d and device %d\n",
                platform_index, device_index);
        }

        if (context == NULL || command_queue == NULL) {

            printf("failed to restore context and command queue for platform %d, device %d\n",
                platform_index, device_index);
            printf("%lu %lu\n", context, command_queue);
        }

        // choose the kernel

        if (device_type == UTC_OCL_DEVICE_GPU) {

            for (kernel_index = 0; kernel_index <= 2; kernel_index++) {

                cl_kernel kernel = utr_ocl_select_kernel(platform_index, device_index, kernel_index, utv_ocl_struct);

                if (monitor > UTC_BASIC_INFO) {
                    printf("\n------------******************************************----------------\n");
                    printf("\nExecution: \t3. restoring kernel %d for platform %d and device %d\n",
                        kernel_index, platform_index, device_index);
                }

                if (context == NULL || command_queue == NULL || kernel == NULL) {

                    printf("failed to restore kernel for platform %d, device %d, kernel %d\n",
                        platform_index, device_index, kernel_index);
                    printf("context %lu, command queue %lu, kernel %lu\n",
                        context, command_queue, kernel);
                }

                for (i = 0; i < ROZMIAR; i++) B[i] = 0.0;

                // call routine to perform matrix transposition

                auto start = high_resolution_clock::now();

                mat_transp_host(kernel_index, A, B, n, context, kernel, command_queue);

                auto stop = high_resolution_clock::now();

                auto duration_time = duration_cast<nanoseconds>(stop - start);

                duration<double> durationInSeconds = duration<double>(duration_time.count()) / 1'000'000'000;

                double seconds = durationInSeconds.count();

                std::cout << "Czas wykonania: " << duration_time.count() << " nanosekund\n" << " ( " << seconds << " seconds)" << std::endl;


                // verify result 
                verify_result(A, B);

            }

        }

    } // end loop over devices

    return;
}

///
//  Cleanup any created OpenCL resources
//
void utr_ocl_cleanup(utt_ocl_struct utv_ocl_struct)
{
    int i, j, k;

    for (i = 0; i < utv_ocl_struct.number_of_platforms; i++) {

        for (j = 0; j < utv_ocl_struct.list_of_platforms[i].number_of_devices; j++) {

            if (utv_ocl_struct.list_of_platforms[i].list_of_devices[j].command_queue != 0) {
                clReleaseCommandQueue(
                    utv_ocl_struct.list_of_platforms[i].list_of_devices[j].command_queue);
            }

            for (k = 0; k < UTC_OCL_MAX_NUM_KERNELS; k++) {
                if (utv_ocl_struct.list_of_platforms[i].list_of_devices[j].kernel[k] != 0) {
                    clReleaseKernel(utv_ocl_struct.list_of_platforms[i].list_of_devices[j].kernel[k]);
                }
            }

            for (k = 0; k < UTC_OCL_MAX_NUM_KERNELS; k++) {
                if (utv_ocl_struct.list_of_platforms[i].list_of_devices[j].program[k] != 0) {
                    clReleaseProgram(utv_ocl_struct.list_of_platforms[i].list_of_devices[j].program[k]);
                }
            }

        }

        free(utv_ocl_struct.list_of_platforms[i].list_of_devices);

        if (utv_ocl_struct.list_of_platforms[i].list_of_contexts[0] != 0)
            clReleaseContext(utv_ocl_struct.list_of_platforms[i].list_of_contexts[0]);

        if (utv_ocl_struct.list_of_platforms[i].list_of_contexts[1] != 0)
            clReleaseContext(utv_ocl_struct.list_of_platforms[i].list_of_contexts[1]);

        if (utv_ocl_struct.list_of_platforms[i].list_of_contexts[2] != 0)
            clReleaseContext(utv_ocl_struct.list_of_platforms[i].list_of_contexts[2]);

    }

    free(utv_ocl_struct.list_of_platforms);

}


int main()
{

    int i;
    utt_ocl_struct utv_ocl_struct;
    utv_ocl_struct.current_platform_index = 0;
    utv_ocl_struct.list_of_platforms = NULL;
    utv_ocl_struct.number_of_platforms = 0;

    /*----------------INITIALIZATION PHASE----------------------*/

    // for all operations indicate explicit info messages
    int monitor = UTC_BASIC_INFO + 1;

    // Create OpenCL contexts on all available platforms
    // contexts are stored in table with indices: 0-CPU, 1-GPU, 2-ACCELERATOR
    // table entry is NULL if there is no such context for a given platform
    int number_of_platforms = utr_ocl_create_contexts(UTC_OCL_ALL_PLATFORMS, monitor, utv_ocl_struct);

    if (number_of_platforms > 1) {
        printf("\nMore than one platform in a system! Check whether it's OK with the code and give the proper ID:\n");
        scanf("%d", &i);
        utv_ocl_struct.current_platform_index = i;
    }
    else {
        utv_ocl_struct.current_platform_index = 0;
    }

    int platform_index = utv_ocl_struct.current_platform_index;
    // create command queues on all devices 
    utr_ocl_create_command_queues(platform_index,
        UTC_OCL_ALL_DEVICES, monitor, utv_ocl_struct);


    //printf(platform_index);

/*----------------KERNEL CREATION PHASE----------------------*/

 //auto start = high_resolution_clock::now();

    create_kernels(utv_ocl_struct);

    /*----------------EXECUTION PHASE----------------------*/

    execute_kernels(utv_ocl_struct);

    //auto stop = high_resolution_clock::now();

    //auto duration = duration_cast<nanoseconds>(stop - start);

    //std::cout << "Czas wykonania: " << duration.count() << " nanosekund" << std::endl;


     /*----------------FINALIZATION PHASE----------------------*/

    utr_ocl_cleanup(utv_ocl_struct);


    //int monitor = UTC_BASIC_INFO + 1;
    //printf("Hello\n");
    //getPizza();
    //getA();
    //int number_of_platforms = utr_ocl_create_contexts(UTC_OCL_ALL_PLATFORMS, monitor);

    //getPizza();

    /*int i;

    int monitor = UTC_BASIC_INFO + 1;

#ifdef time_measurments
    t_begin = time_clock();
#endif



    int number_of_platforms = utr_ocl_create_contexts(UTC_OCL_ALL_PLATFORMS, monitor);

    if (number_of_platforms > 1) {
        printf("\nMore than one platform in a system! Check whether it's OK with the code and give the proper ID:\n");
        scanf("%d", &i);
        utv_ocl_struct.current_platform_index = i;
    }
    else {
        utv_ocl_struct.current_platform_index = 0;
    }*/


    /*printf("YRTYRTYRTYRTYRTYRTYRTY\n");
    printf("\nOpenCL - DISPLAY INFORMATION ABOUT SOFTWARE and HARDWARE\n");
    displayInfo();*/

    //int i, j;
    //char* info;
    //size_t infoSize;
    //cl_uint platformCount;
    //cl_platform_id* platforms;
    //const char* attributeNames[5] = { "Name", "Vendor",
    //    "Version", "Profile", "Extensions" };
    //const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
    //    CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
    //const int attributeCount = sizeof(attributeNames) / sizeof(char*);

    //// get platform count
    //clGetPlatformIDs(5, NULL, &platformCount);

    //// get all platforms
    //platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
    //clGetPlatformIDs(platformCount, platforms, NULL);

    //// for each platform print all attributes
    //for (i = 0; i < platformCount; i++) {

    //    printf("\n %d. Platform \n", i + 1);

    //    for (j = 0; j < attributeCount; j++) {

    //        // get platform attribute value size
    //        clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
    //        info = (char*)malloc(infoSize);

    //        // get platform attribute value
    //        clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);

    //        printf("  %d.%d %-11s: %s\n", i + 1, j + 1, attributeNames[j], info);
    //        free(info);

    //    }

    //    printf("\n");

    //}

    //free(platforms);
    return 0;
}



void displayInfo()
{
    int i, j;
    cl_int retval;
    char* info; size_t size;

    /* POBRANIE I WYSWIETLENIE LICZBY PLATFORM */
    // First, query the total number of platforms
    cl_uint numPlatforms;
    retval = clGetPlatformIDs(0, (cl_platform_id*)NULL, &numPlatforms);
    printf("\nNumber of platforms: %u\n", numPlatforms);

    /* POBRANIE I WYSWIETLENIE INFORMACJI O PLATFORMACH */
    // Next, allocate memory for the installed plaforms, and qeury 
    // to get the list.
    cl_platform_id* platformIds;
    platformIds = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);

    // Then, query the platform IDs
    retval = clGetPlatformIDs(numPlatforms, platformIds, NULL);

    // Iterate through the list of platforms displaying associated information
    printf("CCCCCCCCCCCCCCCC %d", &numPlatforms);
    for (i = 0; i < numPlatforms; i++) {

        printf("\nPlatform ID - %d\n", i);

        //Nazwa
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 0, NULL, &size);
        info = (char*)malloc(size * sizeof(char));
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, size, info, NULL);
        printf("\nPlatform name: ---------------------------------- %s", info);
        free(info);

        //Nazwa producenta
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
        info = (char*)malloc(size * sizeof(char));
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, size, info, NULL);
        printf("\nVendor name: ------------------------------------ %s", info);
        free(info);

        //Wersja platformy
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, 0, NULL, &size);
        info = (char*)malloc(size * sizeof(char));
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, size, info, NULL);
        printf("\nVersion: ---------------------------------------- %s", info);
        free(info);

        //Informacje o profilu wsparcie opencl
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, 0, NULL, &size);
        info = (char*)malloc(size * sizeof(char));
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, size, info, NULL);
        printf("\nPlatform profile: ------------------------------- %s:", info);
        free(info);

        /* POBRANIE I WYSWIETLENIE LISTY URZADZEN */
        cl_uint numDevices;
        retval = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        printf("\n\tNumber of devices: %u\n", numDevices);

        /* POBRANIE INFORMACJI O URZADZENIACH */
        cl_device_id* devicesIds;
        devicesIds = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        retval = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, numDevices, devicesIds, NULL);

        printf("TTYHTYTYHTYHTYHTYH %d", &numDevices);
        for (j = 0; j < numDevices; j++) {

            //Nazwa
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_NAME, 0, NULL, &size);
            info = (char*)malloc(size * sizeof(char));
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_NAME, size, info, &size);
            printf("\nDevice name: ------------------------------------ %s", info);
            free(info);

            //Dostawca
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_VENDOR, 0, NULL, &size);
            info = (char*)malloc(size * sizeof(char));
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_VENDOR, size, info, &size);
            printf("\nDevice vendor: ---------------------------------- %s", info);
            free(info);

            //Dostepny sterownik OpenCL
            retval = clGetDeviceInfo(devicesIds[j], CL_DRIVER_VERSION, 0, NULL, &size);
            info = (char*)malloc(size * sizeof(char));
            retval = clGetDeviceInfo(devicesIds[j], CL_DRIVER_VERSION, size, info, &size);
            printf("\nDriver version: --------------------------------- %s", info);
            free(info);

            //Wspierany OpenCL
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_VERSION, 0, NULL, &size);
            info = (char*)malloc(size * sizeof(char));
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_VERSION, size, info, &size);
            printf("\nSupported OpenCL by device: --------------------- %s", info);
            free(info);

            //Type
            cl_device_type infoType;
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &infoType, NULL);
            switch (infoType)
            {
            case CL_DEVICE_TYPE_CPU:
                printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_CPU");
                break;

            case CL_DEVICE_TYPE_GPU:
                printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_GPU");
                break;

            case CL_DEVICE_TYPE_ACCELERATOR:
                printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_ACCELERATOR");
                break;

            default:
                printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_DEFAULT");
                break;
            };


            //Profil urzadzenia

            //Maksymalna czestotliwosc zegara MHz

            //Rozmiar pamieci globalnej

            //Rozmiar cache pamieci globalnej

            //Rozmiar linijki cache pamieci globalnej

            //Typ pamieci lokalnej

            //Rozmiar pamieci lokalnej

            //Maksymalny rozmiar pamięci do zaalokowania

            //Maksymalny rozmiar bufora stałych

            //Maksymalna liczba jednostek obliczeniowych

            //Maksymalny rozmiar grupy roboczej

            //Maksymalny wymiar przestrzeni wątków

            //Rozszerzenia


        }
        free(devicesIds);

        printf("\n\n\t ---- ************* ----\n");
    }
    printf("\n\n");
    free(platformIds);

    // create contexts for all platforms
    //int platform_id = -1; // according to convention -1 means all platforms
    //int monitor = 3;  // according to convention - print all info
    //utr_ocl_create_contexts(platform_id, monitor);

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
