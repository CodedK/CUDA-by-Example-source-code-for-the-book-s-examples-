/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "../common/book.h"

#define SIZE    (100*1024*1024)

int main( void ) {
    unsigned char *buffer =
                     (unsigned char*)big_random_block( SIZE );

    // capture the start time
    clock_t         start, stop;
    start = clock();

    unsigned int    histo[256];
    for (int i=0; i<256; i++)
        histo[i] = 0;

    for (int i=0; i<SIZE; i++)
        histo[buffer[i]]++;

    stop = clock();
    float   elapsedTime = (float)(stop - start) /
                          (float)CLOCKS_PER_SEC * 1000.0f;
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    long histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += histo[i];
    }
    printf( "Histogram Sum:  %ld\n", histoCount );

    free( buffer );
    return 0;
}
