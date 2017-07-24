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


#ifndef __CPU_ANIM_H__
#define __CPU_ANIM_H__

#include "gl_helper.h"

#include <iostream>


struct CPUAnimBitmap {
    unsigned char    *pixels;
    int     width, height;
    void    *dataBlock;
    void (*fAnim)(void*,int);
    void (*animExit)(void*);
    void (*clickDrag)(void*,int,int,int,int);
    int     dragStartX, dragStartY;

    CPUAnimBitmap( int w, int h, void *d = NULL ) {
        width = w;
        height = h;
        pixels = new unsigned char[width * height * 4];
        dataBlock = d;
        clickDrag = NULL;
    }

    ~CPUAnimBitmap() {
        delete [] pixels;
    }

    unsigned char* get_ptr( void ) const   { return pixels; }
    long image_size( void ) const { return width * height * 4; }

    void click_drag( void (*f)(void*,int,int,int,int)) {
        clickDrag = f;
    }

    void anim_and_exit( void (*f)(void*,int), void(*e)(void*) ) {
        CPUAnimBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        fAnim = f;
        animExit = e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = "";
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
        glutInitWindowSize( width, height );
        glutCreateWindow( "bitmap" );
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        if (clickDrag != NULL)
            glutMouseFunc( mouse_func );
        glutIdleFunc( idle_func );
        glutMainLoop();
    }

    // static method used for glut callbacks
    static CPUAnimBitmap** get_bitmap_ptr( void ) {
        static CPUAnimBitmap*   gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func( int button, int state,
                            int mx, int my ) {
        if (button == GLUT_LEFT_BUTTON) {
            CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            } else if (state == GLUT_UP) {
                bitmap->clickDrag( bitmap->dataBlock,
                                   bitmap->dragStartX,
                                   bitmap->dragStartY,
                                   mx, my );
            }
        }
    }

    // static method used for glut callbacks
    static void idle_func( void ) {
        static int ticks = 1;
        CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
        bitmap->fAnim( bitmap->dataBlock, ticks++ );
        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27:
                CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
                bitmap->animExit( bitmap->dataBlock );
                //delete bitmap;
                exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw( void ) {
        CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
        glutSwapBuffers();
    }
};


#endif  // __CPU_ANIM_H__

