# Introduction

This is a Linux compilation aggregation for the OpenGL Cuda interop example written by allanmac -- https://devtalk.nvidia.com/member/1794023/ . I don't claim copyright for any of the code in this repo which belongs to their respective authors. All I have done is building the scons file, and fixed compilation under my Linux environment.

# Sources referenced

The following sources were referenced:

    * https://bitbucket.org/scons/scons/wiki/CudaTool
    * https://devtalk.nvidia.com/default/topic/816415/opengl/glfw-cuda-amp-opengl-interop-example-code/post/4507382/#4507382: "I have a very simple OpenGL/CUDA + GLFW interop example here."
    
# Building

Additional commands used in creating of this repo:

    git clone https://gist.github.com/allanmac/4ff11985c3562830989f
    cd 4ff11985c3562830989f
    pip install glad
    python -m glad --spec gl --generator c --out-path ./glad
    scons
