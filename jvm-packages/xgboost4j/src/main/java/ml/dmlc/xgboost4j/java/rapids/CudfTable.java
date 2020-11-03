package ml.dmlc.xgboost4j.java.rapids;

import java.util.Arrays;

import java.util.List;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;

import com.google.common.primitives.Ints;

/**
 * The base cudf table.
 */
public class CudfTable implements AutoCloseable {
  private Table table;
  private int[] featureIndices;
  private int[] labelIndices;
  private int[] weightIndices;
  private int[] baseMarginIndices;

  // For Test
  public CudfTable(Table table) {
    this.table = table;
  }

  public CudfTable(Table table, List<Integer> featureIndices, List<Integer> labelIndices,
                   List<Integer> weightIndices, List<Integer> marginIndices) {
    this(table,
      Ints.toArray(featureIndices),
      Ints.toArray(labelIndices),
      weightIndices != null && !weightIndices.isEmpty() ? Ints.toArray(weightIndices) : null,
      marginIndices != null && !marginIndices.isEmpty() ? Ints.toArray(marginIndices) : null);
  }

  public CudfTable(Table table, int[] featureIndices, int[] labelIndices, int[] weightIndices,
                   int[] baseMarginIndices) {
    this.table = table;
    this.featureIndices = featureIndices;
    this.labelIndices = labelIndices;
    this.weightIndices = weightIndices;
    this.baseMarginIndices = baseMarginIndices;

    validate();
  }

  private void validate() {
    if (labelIndices == null) {
      throw new RuntimeException("should specify label column");
    }

    if (featureIndices == null) {
      throw new RuntimeException("should specify feature columns");
    }
  }

  public void setTable(Table table) {
    this.table = table;
  }
  public Table getTable() {
    return table;
  }

  public long getNumRows() {
    return table.getRowCount();
  }

  public int getNumColumns() {
    return table.getNumberOfColumns();
  }

  public ColumnVector getColumnVector(int index) {
    return table.getColumn(index);
  }

  public String getFeatureArrayInterface() {
    return getArrayInterface(featureIndices);
  }

  public String getLabelArrayInterface() {
    return getArrayInterface(labelIndices);
  }

  public String getWeightArrayInterface() {
    return getArrayInterface(weightIndices);
  }

  public String getBaseMarginArrayInterface() {
    return getArrayInterface(baseMarginIndices);
  }

  public String getArrayInterface(int... indices) {
    if (indices == null || indices.length == 0) return "";
    return ColumnData.getArrayInterface(getAsColumnData(indices));
  }

  public ColumnData[] getAsColumnData(int... indices) {
    if (indices == null || indices.length == 0) return new ColumnData[]{};
    return Arrays.stream(indices)
      .mapToObj(this::getColumnVector)
      .map(this::getColumnData)
      .toArray(ColumnData[]::new);
  }

  private ColumnData getColumnData(ColumnVector columnVector) {
    BaseDeviceMemoryBuffer dataBuffer = columnVector.getDeviceBufferFor(BufferType.DATA);
    BaseDeviceMemoryBuffer validBuffer = columnVector.getDeviceBufferFor(BufferType.VALIDITY);
    long validPtr = 0;
    if (validBuffer != null) {
      validPtr = validBuffer.getAddress();
    }
    DType dType = columnVector.getType();
    String typeStr = "";
    if (dType == DType.FLOAT32 || dType == DType.FLOAT64 ||
        dType == DType.TIMESTAMP_DAYS || dType == DType.TIMESTAMP_MICROSECONDS ||
        dType == DType.TIMESTAMP_MILLISECONDS || dType == DType.TIMESTAMP_NANOSECONDS ||
        dType == DType.TIMESTAMP_SECONDS) {
      typeStr = "<f" + dType.getSizeInBytes();
    } else if (dType == DType.BOOL8 || dType == DType.INT8 || dType == DType.INT16 ||
        dType == DType.INT32 || dType == DType.INT64) {
      typeStr = "<i" + dType.getSizeInBytes();
    } else {
        // not supporting type.
    }

    return new ColumnData(dataBuffer.getAddress(), columnVector.getRowCount(), validPtr,
      dType.getSizeInBytes(), typeStr, columnVector.getNullCount());
  }

  public String getCudfJsonStr() {
    StringBuilder builder = new StringBuilder();

    builder.append("{");

    String featureStr = getFeatureArrayInterface();
    if (featureStr == null || featureStr.isEmpty()) {
      throw new RuntimeException("Feature json should not be empty");
    } else {
      builder.append("\"features_str\":" + featureStr);
    }

    String labelStr = getLabelArrayInterface();
    if (labelStr == null || labelStr.isEmpty()) {
      throw new RuntimeException("label json should not be empty");
    } else {
      builder.append(",\"label_str\":" + labelStr);
    }

    String weightStr = getWeightArrayInterface();
    if (weightStr != null && ! weightStr.isEmpty()) {
      builder.append(",\"weight_str\":" + weightStr);
    }

    String baseMarginStr = getBaseMarginArrayInterface();
    if (baseMarginStr != null && ! baseMarginStr.isEmpty()) {
      builder.append(",\"basemargin_str\":" + baseMarginStr);
    }

    builder.append("}");

    return builder.toString();
  }

  @Override
  public void close() throws Exception {
    if (table != null) table.close();
  }
}
