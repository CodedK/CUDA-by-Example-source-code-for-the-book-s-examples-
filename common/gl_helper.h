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


#ifndef __GL_HELPER_H__
#define __GL_HELPER_H__

/*
   On 64-bit Windows, we need to prevent GLUT from automatically linking against
   glut32. We do this by defining GLUT_NO_LIB_PRAGMA. This means that we need to
   manually add opengl32.lib and glut64.lib back to the link using pragmas.
   Alternatively, this could be done on the compilation/link command-line, but
   we chose this so that compilation is identical between 32- and 64-bit Windows.
*/
#ifdef _WIN64
#define GLUT_NO_LIB_PRAGMA
#pragma comment (lib, "opengl32.lib")  /* link with Microsoft OpenGL lib */
#pragma comment (lib, "glut64.lib")    /* link with Win64 GLUT lib */
#endif //_WIN64


#ifdef _WIN32
/* On Windows, include the local copy of glut.h and glext.h */
#include "GL/glut.h"
#include "GL/glext.h"

#define GET_PROC_ADDRESS( str ) wglGetProcAddress( str )

#else

/* On Linux, include the system's copy of glut.h, glext.h, and glx.h */
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>

#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )

#endif //_WIN32


#endif //__GL_HELPER_H__'
