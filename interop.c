
//
//
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <stdlib.h>

//
//
//

#include "interop.h"

//
//
//

struct pxl_interop
{
  int                     count;
  int                     index;

  // w x h
  int                     width;
  int                     height;

  // GL buffers
  GLuint*                 fb;
  GLuint*                 rb;

  // CUDA resources
  cudaGraphicsResource_t* cgr;
  cudaArray_t*            ca;

  // CUDA stream
  cudaStream_t            stream;
};

//
//
//

struct pxl_interop*
pxl_interop_create(const int fbo_count)
{
  struct pxl_interop* const interop = calloc(1,sizeof(*interop));

  interop->count = fbo_count;
  
  // allocate arrays
  interop->fb  = calloc(fbo_count,sizeof(*(interop->fb )));
  interop->rb  = calloc(fbo_count,sizeof(*(interop->rb )));
  interop->cgr = calloc(fbo_count,sizeof(*(interop->cgr)));
  interop->ca  = calloc(fbo_count,sizeof(*(interop->ca)));

  // render buffer object w/a color buffer
  glCreateRenderbuffers(fbo_count,interop->rb);

  // frame buffer object
  glCreateFramebuffers(fbo_count,interop->fb);

  // attach rb0 to the fb0
  for (int index=0; index<fbo_count; index++)  
    {
      glNamedFramebufferRenderbuffer(interop->fb[index],
                                     GL_COLOR_ATTACHMENT0,
                                     GL_RENDERBUFFER,
                                     interop->rb[index]);
    }

  // create an interop stream 
  cudaError_t cuda_err = cudaStreamCreate(&interop->stream);

  // return it
  return interop;
}


void
pxl_interop_destroy(struct pxl_interop* const interop)
{
  // resize color buffer
  for (int index=0; index<interop->count; index++)
    {
      // unregister CUDA resource
      if (interop->cgr[index] != NULL)
        cudaGraphicsUnregisterResource(interop->cgr[index]);
    }

  // destroy stream
  cudaStreamDestroy(interop->stream);

  // delete rbo's
  glDeleteRenderbuffers(interop->count,interop->rb);

  // delete fbo's
  glDeleteFramebuffers(interop->count,interop->fb);

  // free interop
  free(interop);
}

//
//
//

cudaError_t
pxl_interop_size_set(struct pxl_interop* const interop, const int width, const int height)
{
  cudaError_t cuda_err = cudaSuccess;

  // save new size
  interop->width  = width;
  interop->height = height;

  // resize color buffer
  for (int index=0; index<interop->count; index++)
    {
      // unregister resource
      if (interop->cgr[index] != NULL)
        cuda_err = cudaGraphicsUnregisterResource(interop->cgr[index]);

      // resize rbo
      glNamedRenderbufferStorage(interop->rb[index],GL_RGBA8,width,height);

      // cudaGraphicsResource_t cgr;      

      // register rbo
      cuda_err = cudaGraphicsGLRegisterImage(&interop->cgr[index],
                                             interop->rb[index],
                                             GL_RENDERBUFFER,
                                             cudaGraphicsRegisterFlagsSurfaceLoadStore | 
                                             cudaGraphicsRegisterFlagsWriteDiscard);

      // map graphics resource
      cuda_err = cudaGraphicsMapResources(1,&interop->cgr[index],interop->stream);

      // get a CUDA Array
      cuda_err = cudaGraphicsSubResourceGetMappedArray(&interop->ca[index],
                                                       interop->cgr[index],
                                                       0,0);

      // unmap graphics resource
      cuda_err = cudaGraphicsUnmapResources(1,&interop->cgr[index],interop->stream);

      // probe fbo status
      glCheckNamedFramebufferStatus(interop->fb[index],0);
    }
  
  return cuda_err;
}

void
pxl_interop_size_get(struct pxl_interop* const interop, int* const width, int* const height)
{
  *width  = interop->width;
  *height = interop->height;
}

//
//
//

cudaArray_const_t
pxl_interop_array_get(struct pxl_interop* const interop)
{
  return interop->ca[interop->index];
}

cudaStream_t
pxl_interop_stream_get(struct pxl_interop* const interop)
{
  return interop->stream;
}

//
//
//

void
pxl_interop_swap(struct pxl_interop* const interop)
{
  interop->index = (interop->index + 1) % interop->count;
}

//
//
//

void
pxl_interop_clear(struct pxl_interop* const interop)
{
  /*
  const GLenum draw_buffer[] = { GL_COLOR_ATTACHMENT0 };
  const GLuint clear_color[] = { 255, 0, 0, 255 };
                       
  glNamedFramebufferDrawBuffers(interop->fb0,1,draw_buffer);
  glClearNamedFramebufferuiv(interop->fb0,GL_COLOR,0,clear_color);
  */

  static const GLenum attachments[] = { GL_COLOR_ATTACHMENT0 };

  glInvalidateNamedFramebufferData(interop->fb[interop->index],1,attachments);
}

//
//
//

void
pxl_interop_blit(struct pxl_interop* const interop)
{
  glBlitNamedFramebuffer(interop->fb[interop->index],0,
                         0,0,              interop->width,interop->height,
                         0,interop->height,interop->width,0,
                         GL_COLOR_BUFFER_BIT,
                         GL_NEAREST);
}

//
//
//
