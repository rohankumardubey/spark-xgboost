/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudf.h>
#include <rmm/rmm.h>

#include "xgboost4j_spark_gpu.h"
#include "xgboost4j_spark.h"

namespace xgboost {
namespace spark {

/*! \brief utility class to track GPU allocations */
class unique_gpu_ptr {
  void* ptr;

public:
  unique_gpu_ptr(unique_gpu_ptr const&) = delete;
  unique_gpu_ptr& operator=(unique_gpu_ptr const&) = delete;

  unique_gpu_ptr(unique_gpu_ptr&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }

  unique_gpu_ptr(size_t size) : ptr(nullptr) {
    rmmError_t rmm_status = RMM_ALLOC(&ptr, size, 0);
    if (rmm_status != RMM_SUCCESS) {
      throw std::bad_alloc();
    }
  }

  ~unique_gpu_ptr() {
    if (ptr != nullptr) {
      RMM_FREE(ptr, 0);
    }
  }

  void* get() {
    return ptr;
  }

  void* release() {
    void* result = ptr;
    ptr = nullptr;
    return result;
  }
};

/*! \brief custom deleter to free malloc allocations */
struct malloc_deleter {
  void operator()(void* ptr) const {
    free(ptr);
  }
};

static unsigned int get_unsaferow_nullset_size(unsigned int num_columns) {
  // The nullset size is rounded up to a multiple of 8 bytes.
  return ((num_columns + 63) / 64) * 8;
}

/*! \brief Returns the byte width of the specified gdf_dtype or 0 on error. */
static size_t get_dtype_size(gdf_dtype dtype) {
  switch (dtype) {
  case GDF_BOOL8:
  case GDF_INT8:
    return 1;
  case GDF_INT16:
    return 2;
  case GDF_INT32:
  case GDF_FLOAT32:
  case GDF_DATE32:
    return 4;
  case GDF_INT64:
  case GDF_FLOAT64:
  case GDF_DATE64:
  case GDF_TIMESTAMP:
    return 8;
  default:
    break;
  }
  return 0;
}

static void build_unsafe_row_nullsets(void* unsafe_rows_dptr,
    std::vector<gdf_column const*> const& gdfcols) {
  unsigned int num_columns = gdfcols.size();
  size_t num_rows = gdfcols[0]->size;

  // make the array of validity data pointers available on the device
  std::vector<uint8_t const*> valid_ptrs(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    valid_ptrs[i] = gdfcols[i]->valid;
  }
  unique_gpu_ptr dev_valid_mem(num_columns * sizeof(*valid_ptrs.data()));
  uint8_t** dev_valid_ptrs = reinterpret_cast<uint8_t**>(dev_valid_mem.get());
  cudaError_t cuda_status = cudaMemcpy(dev_valid_ptrs, valid_ptrs.data(),
      num_columns * sizeof(valid_ptrs[0]), cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_status));
  }

  // build the nullsets for each UnsafeRow
  cuda_status = xgboost::spark::build_unsaferow_nullsets(
      reinterpret_cast<uint64_t*>(unsafe_rows_dptr), dev_valid_ptrs,
      num_columns, num_rows);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_status));
  }
}

/*!
 * \brief Transforms a set of cudf columns into an array of Spark UnsafeRow.
 * NOTE: Only fixed-length datatypes are supported, as it is assumed
 * that every UnsafeRow has the same size.
 *
 * Spark's UnsafeRow with fixed-length datatypes has the following format:
 *   null bitset, 8-byte value, [8-byte value, ...]
 * where the null bitset is a collection of 64-bit words with each bit
 * indicating whether the corresponding field is null.
 */
void* build_unsafe_rows(std::vector<gdf_column const*> const& gdfcols) {
  cudaError_t cuda_status;
  unsigned int num_columns = gdfcols.size();
  size_t num_rows = gdfcols[0]->size;
  unsigned int nullset_size = get_unsaferow_nullset_size(num_columns);
  unsigned int row_size = nullset_size + num_columns * 8;
  size_t unsafe_rows_size = num_rows * row_size;

  // allocate GPU memory to hold the resulting UnsafeRow array
  unique_gpu_ptr unsafe_rows_devmem(unsafe_rows_size);
  uint8_t* unsafe_rows_dptr = static_cast<uint8_t*>(unsafe_rows_devmem.get());

  // write each column to the corresponding position in the unsafe rows
  for (int i = 0; i < num_columns; ++i) {
    // point to the corresponding field in the first UnsafeRow
    uint8_t* dest_addr = unsafe_rows_dptr + nullset_size + i * 8;
    unsigned int dtype_size = get_dtype_size(gdfcols[i]->dtype);
    if (dtype_size == 0) {
      throw std::runtime_error("Unsupported column type");
    }

    cuda_status = xgboost::spark::store_with_stride_async(dest_addr,
        gdfcols[i]->data, num_rows, dtype_size, row_size, 0);
    if (cuda_status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(cuda_status));
    }
  }

  build_unsafe_row_nullsets(unsafe_rows_dptr, gdfcols);

  // copy UnsafeRow results back to host
  std::unique_ptr<void, malloc_deleter> unsafe_rows(malloc(unsafe_rows_size));
  if (unsafe_rows.get() == nullptr) {
    throw std::bad_alloc();
  }
  // This copy also serves as a synchronization point with the GPU.
  cuda_status = cudaMemcpy(unsafe_rows.get(), unsafe_rows_dptr,
      unsafe_rows_size, cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_status));
  }

  return unsafe_rows.release();
}

} // namespace spark
} // namespace xgboost


static void throw_java_exception(JNIEnv* env, char const* classname,
    char const* msg) {
  jclass exClass = env->FindClass(classname);
  if (exClass != NULL) {
    env->ThrowNew(exClass, msg);
  }
}

static void throw_java_exception(JNIEnv* env, char const* msg) {
  throw_java_exception(env, "java/lang/RuntimeException", msg);
}

JNIEXPORT jlong JNICALL
Java_ml_dmlc_xgboost4j_java_XGBoostSparkJNI_buildUnsafeRows(JNIEnv * env,
    jclass clazz, jlongArray nativeColumnPtrs) {
  int num_columns = env->GetArrayLength(nativeColumnPtrs);
  if (env->ExceptionOccurred()) {
    return 0;
  }
  if (num_columns <= 0) {
    throw_java_exception(env, "Invalid number of columns");
    return 0;
  }

  std::vector<gdf_column const*> gdfcols(num_columns);
  jlong* column_jlongs = env->GetLongArrayElements(nativeColumnPtrs, nullptr);
  if (column_jlongs == nullptr) {
    return 0;
  }
  for (int i = 0; i < num_columns; ++i) {
    gdfcols[i] = reinterpret_cast<gdf_column*>(column_jlongs[i]);
  }
  env->ReleaseLongArrayElements(nativeColumnPtrs, column_jlongs, JNI_ABORT);

  void* unsafe_rows = nullptr;
  try {
    unsafe_rows = xgboost::spark::build_unsafe_rows(gdfcols);
  } catch (std::bad_alloc const& e) {
    throw_java_exception(env, "java/lang/OutOfMemoryError",
        "Could not allocate native memory");
  } catch (std::exception const& e) {
    throw_java_exception(env, e.what());
  }

  return reinterpret_cast<jlong>(unsafe_rows);
}
