#ifndef __HIP_ERROR_CHECK_H
#define __HIP_ERROR_CHECK_H

#define HIP_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __hipSafeCall( hipError_t err, const char *file, const int line )
{
#ifdef HIP_ERROR_CHECK
    if ( hipSuccess != err )
    {
        fprintf( stderr, "hipSafeCall() failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#endif
