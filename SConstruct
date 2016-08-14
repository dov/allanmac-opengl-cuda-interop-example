# create a SCons environment
env = Environment()
env['CUDA_TOOLKIT_PATH'] = '/usr/local/cuda-7.5'
env['CUDA_SDK_PATH'] = '/usr/local/cuda-7.5/targets/x86_64-linux'

# direct SCons to the nvcc.py script
env.Tool('cuda', toolpath = ['scons-tools'])
env.Append(NVCCFLAGS=['-I../include',
                       '-std=c++11',
                       '-ccbin=/usr/local/gcc-5.4.0/bin/g++'
                       ],
           CPPPATH = ['./glad/include'],
           LIBS=['GL','GLU', 'glut','GLEW','glfw','m'],
           CPLUSPLUS='/usr/local/gcc-5.4.0/bin/g++')
env['CXX'] = '/usr/local/gcc-5.4.0/bin/g++'
env['CC'] = '/usr/local/gcc-5.4.0/bin/gcc'
env['LINK'] = '/usr/local/gcc-5.4.0/bin/g++'
env.Append(LIBS=["GL","GLU", "glut","GLEW",'dl'])


# now create the simpleGL program
env.Program('test-cuda-opengl',
            ['main.c',
             'interop.c',
             'assert_cuda.c',
             'kernel.cu',
             'glad/src/glad.c'])
