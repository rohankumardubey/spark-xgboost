/*!
 * Copyright 2018 by xgboost contributors
 */

namespace xgboost {
namespace data {

/**
 * Convert the data element into a common format
 * the dtype should match https://github.com/rapidsai/cudf/blob/branch-0.15/java/src/main/java/ai/rapids/cudf/DType.java#L20
 */
__device__ inline float ConvertDataElement(void const* data, int tid, int dtype) {
  switch(dtype) {
    case 1: { //int8
      int8_t * d = (int8_t*)data;
      return float(d[tid]);
    }
    case 2: { //INT16
      int16_t * d = (int16_t*)data;
      return float(d[tid]);
    }
    case 3: { //INT32
      int32_t * d = (int32_t*)data;
      return float(d[tid]);
    }
    case 4: { //INT64
      int64_t * d = (int64_t*)data;
      return float(d[tid]);
    }
    case 9: { //FLOAT32
      float * d = (float *)data;
      return float(d[tid]);
    }
    case 10: { //FLOAT64
      double * d = (double *)data;
      return float(d[tid]);
    }
  }
  return nanf(nullptr);
}

}  // namespace data
}  // namespace xgboost
