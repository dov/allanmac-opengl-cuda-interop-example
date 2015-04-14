//
//
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>

//
//
//

#include <stdlib.h>
#include <stdio.h>

//
//
//

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//
//
//

#include "interop.h"

//
// FPS COUNTER FROM HERE:
//
// http://antongerdelan.net/opengl/glcontext2.html
//

static
void
pxl_glfw_fps(GLFWwindow* window)
{
  // static fps counters
  static double stamp_prev  = 0.0;
  static int    frame_count = 0;

  // locals
  const double stamp_curr = glfwGetTime();
  const double elapsed    = stamp_curr - stamp_prev;
  
  if (elapsed > 0.5)
    {
      stamp_prev = stamp_curr;
      
      const double fps = (double)frame_count / elapsed;

      int  width, height;
      char tmp[64];

      glfwGetFramebufferSize(window,&width,&height);
  
      sprintf_s(tmp,64,"(%u x %u) - FPS: %.2f",width,height,fps);

      glfwSetWindowTitle(window,tmp);

      frame_count = 0;
    }

  frame_count++;
}

//
//
//

static
void
pxl_glfw_error_callback(int error, const char* description)
{
  fputs(description,stderr);
}

static
void
pxl_glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
}

static
void
pxl_glfw_init(GLFWwindow** window, const int width, const int height)
{
  //
  // INITIALIZE GLFW/GLAD
  //
  
  glfwSetErrorCallback(pxl_glfw_error_callback);

  if (!glfwInit())
    exit(EXIT_FAILURE);
 
  const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

  glfwWindowHint(GLFW_DEPTH_BITS,            0);
  glfwWindowHint(GLFW_STENCIL_BITS,          0);

  glfwWindowHint(GLFW_SRGB_CAPABLE,          GL_TRUE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

  glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);

  *window = glfwCreateWindow(width,height,"GLFW / CUDA Interop",NULL,NULL);

  if (*window == NULL)
    {
      glfwTerminate();
      exit(EXIT_FAILURE);
    }

  glfwMakeContextCurrent(*window);

  // set up GLAD
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

  // ignore vsync for now
  glfwSwapInterval(0);

  // enable SRGB 
  glEnable(GL_FRAMEBUFFER_SRGB);

  // only copy r/g/b
  glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_FALSE);
}

//
//
//

static
void
pxl_glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
  // get context
  struct pxl_interop* const interop = glfwGetWindowUserPointer(window);

  pxl_interop_resize(interop,width,height);
}

//
//
//

cudaError_t
pxl_kernel_launcher(cudaArray_const_t array, const int width, const int height, cudaStream_t stream);

//
//
//

int
main(int argc, char* argv[])
{
  //
  // INIT GLFW
  //

  GLFWwindow* window;

  pxl_glfw_init(&window,1024,1024);

  //
  // SET CALLBACKS
  //

  glfwSetKeyCallback            (window,pxl_glfw_key_callback);
  glfwSetFramebufferSizeCallback(window,pxl_glfw_window_size_callback);
  
  //
  // INIT CUDA
  //
  cudaError_t cuda_err;
  
  int gl_device_id,gl_device_count;
      
  cuda_err = cudaGLGetDevices(&gl_device_count,&gl_device_id,1,cudaGLDeviceListAll);

  int cuda_device_id = (argc > 1) ? atoi(argv[1]) : gl_device_id;
  
  cuda_err = cudaSetDevice(cuda_device_id);

  //
  // INFO
  //
  struct cudaDeviceProp props;

  cuda_err = cudaGetDeviceProperties(&props,gl_device_id);
  printf("GL   : %-24s (%2d)\n",props.name,props.multiProcessorCount);

  cuda_err = cudaGetDeviceProperties(&props,cuda_device_id);
  printf("CUDA : %-24s (%2d)\n",props.name,props.multiProcessorCount);

  //
  // CREATE A CUDA STREAM
  //

  cudaStream_t stream;
  
  cuda_err = cudaStreamCreate(&stream);

  //
  // CREATE AND SAVE INTEROP INSTANCE
  //
  
  struct pxl_interop* const interop = pxl_interop_create(window);

  glfwSetWindowUserPointer(window,interop);

  //
  // GET ACTUAL WINDOW SIZE
  //
  
  int width, height;

  // get initial width/height
  glfwGetFramebufferSize(window,&width,&height);

  // resize with initial window dimensions
  cuda_err = pxl_interop_resize(interop,width,height);

  //
  // LOOP UNTIL DONE
  //
  
  while (!glfwWindowShouldClose(window))
    {
      //
      // MONITOR FPS
      //

      pxl_glfw_fps(window);
      
      //
      // EXECUTE CUDA KERNEL ON RENDER BUFFER
      //

      int         width,height;
      cudaArray_t cuda_array;

      pxl_interop_get_size(interop,&width,&height);

      cuda_err = pxl_interop_map(interop,&cuda_array,stream);

      cuda_err = pxl_kernel_launcher(cuda_array,width,height,stream);

      cuda_err = pxl_interop_unmap(interop,stream);

      cuda_err = cudaStreamSynchronize(stream);

      //
      // BLIT
      // 

      pxl_interop_blit(interop);

      //
      // SWAP
      //
      
      glfwSwapBuffers(window);

      //
      // PUMP/POLL/WAIT
      //
      
      glfwPollEvents(); // glfwWaitEvents();
    }

  //
  // CLEANUP
  //
  
  pxl_interop_destroy(interop);
  
  glfwDestroyWindow(window);

  glfwTerminate();

  cudaDeviceReset();

  // missing some clean up here
  
  exit(EXIT_SUCCESS);
}

//
//
//