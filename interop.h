
/*
 * Copyright 2015 Allan MacKinnon.  All rights reserved.
 *
 */

#pragma once

//
//
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>

//
//
//

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//
//
//

struct pxl_interop*
pxl_interop_create();

void
pxl_interop_destroy(struct pxl_interop* const interop);

//
//
//

cudaError_t
pxl_interop_resize(struct pxl_interop* const interop, const int width, const int height);

void
pxl_interop_get_size(struct pxl_interop* const interop, int* const width, int* const height);

//
//
//

cudaError_t
pxl_interop_map(struct pxl_interop* const interop, cudaArray_t* cuda_array, cudaStream_t stream);

cudaError_t
pxl_interop_unmap(struct pxl_interop* const interop, cudaStream_t stream);

//
//
//

void
pxl_interop_clear(struct pxl_interop* const interop);

void
pxl_interop_blit(struct pxl_interop* const interop);

//
//
//