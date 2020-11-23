/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package ml.dmlc.xgboost4j.java;

import com.google.common.primitives.Floats;

import junit.framework.TestCase;

import org.apache.commons.lang.ArrayUtils;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;
import ml.dmlc.xgboost4j.java.rapids.CudfTable;

/**
 * test cases for DMatrix
 */
public class DMatrixForArrayInterfaceTest {
  @Test
  public void testCreateFromArrayInterfaceColumns() {
    Float[] labelFloats = new Float[]{2f, 4f, 6f, 8f, 10f};
    Table table = new Table.TestBuilder()
      .column(1.f, null, 5.f, 7.f, 9.f)
      .column(labelFloats)
      .build();

    try {
      CudfTable cudfTable = new CudfTable(table, new int[]{0}, new int[]{1}, null, null);
      String featureJson = cudfTable.getFeatureArrayInterface();
      String anchorJson = cudfTable.getLabelArrayInterface();

      try {
        DMatrix dMatrix;
        dMatrix = new DMatrix(featureJson, 0, 1);
        dMatrix.setLabel(anchorJson);
        dMatrix.setWeight(anchorJson);
        dMatrix.setBaseMargin(anchorJson);
        float[] anchor = convertFloatTofloat(labelFloats);
        float[] label = dMatrix.getLabel();
        float[] weight = dMatrix.getWeight();
        float[] baseMargin = dMatrix.getBaseMargin();
        TestCase.assertTrue(Arrays.equals(anchor, label));
        TestCase.assertTrue(Arrays.equals(anchor, weight));
        TestCase.assertTrue(Arrays.equals(anchor, baseMargin));
      } catch (XGBoostError xgBoostError) {
        xgBoostError.printStackTrace();
      }
    } finally {
      if (table != null) table.close();
    }
  }

  @Test
  public void testCreateFromColumnDataIterator() throws XGBoostError {
    java.util.List<CudfTable> rapidsTables = new java.util.LinkedList<CudfTable>();

    Float[] label1 = {25f, 21f, 22f, 20f, 24f};
    Float[] weight1 = {1.3f, 2.31f, 0.32f, 3.3f, 1.34f};
    Float[] baseMargin1 = {1.2f, 0.2f, 1.3f, 2.4f, 3.5f};

    Table table = new Table.TestBuilder()
      .column(1.2f, null, 5.2f, 7.2f, 9.2f)
      .column(0.2f, 0.4f, 0.6f, 2.6f, 0.10f)
      .column(label1)
      .column(weight1)
      .column(baseMargin1)
      .build();

    rapidsTables.add(new CudfTable(table, new int[]{0, 1}, new int[]{2}, new int[]{3}, new int[]{4}));

    Float[] label2 = {9f, 5f, 4f, 10f, 12f};
    Float[] weight2 = {3.0f, 1.3f, 3.2f, 0.3f, 1.34f};
    Float[] baseMargin2 = {0.2f, 2.5f, 3.1f, 4.4f, 2.2f};
    Table table1 = new Table.TestBuilder()
      .column(11.2f, 11.2f, 15.2f, 17.2f, 19.2f)
      .column(1.2f, 1.4f, null, 12.6f, 10.10f)
      .column(label2)
      .column(weight2)
      .column(baseMargin2)
      .build();

    rapidsTables.add(new CudfTable(table1, new int[]{0, 1}, new int[]{2}, new int[]{3}, new int[]{4}));

    DMatrix dmat = new DMatrix(rapidsTables.iterator(), 0.0f, 8, 1);

    float[] anchorLabel = convertFloatTofloat((Float[]) ArrayUtils.addAll(label1, label2));
    float[] anchorWeight = convertFloatTofloat((Float[]) ArrayUtils.addAll(weight1, weight2));
    float[] anchorBaseMargin = convertFloatTofloat((Float[]) ArrayUtils.addAll(baseMargin1, baseMargin2));

    TestCase.assertTrue(Arrays.equals(anchorLabel, dmat.getLabel()));
    TestCase.assertTrue(Arrays.equals(anchorWeight, dmat.getWeight()));
    TestCase.assertTrue(Arrays.equals(anchorBaseMargin, dmat.getBaseMargin()));

    table.close();
    table1.close();
  }

  @Test
  public void testCreateFromColumnDataIteratorWithoutWeight() throws XGBoostError {
    java.util.List<CudfTable> rapidsTables = new java.util.LinkedList<CudfTable>();

    Float[] label1 = {25f, 21f, 22f, 20f, 24f};

    Table table = new Table.TestBuilder()
      .column(1.2f, null, 5.2f, 7.2f, 9.2f)
      .column(0.2f, 0.4f, 0.6f, 2.6f, 0.10f)
      .column(label1)
      .build();

    rapidsTables.add(new CudfTable(table, new int[]{0, 1}, new int[]{2}, null, null));

    Float[] label2 = {9f, 5f, 4f, 10f, 12f};
    Table table1 = new Table.TestBuilder()
      .column(11.2f, 11.2f, 15.2f, 17.2f, 19.2f)
      .column(1.2f, 1.4f, null, 12.6f, 10.10f)
      .column(label2)
      .build();

    rapidsTables.add(new CudfTable(table1, new int[]{0, 1}, new int[]{2}, null, null));

    DMatrix dmat = new DMatrix(rapidsTables.iterator(), 0.0f, 8, 1);

    float[] actualLabel = convertFloatTofloat((Float[]) ArrayUtils.addAll(label1, label2));

    float[] txxx = dmat.getLabel();

    System.out.println("size:" + txxx.length + ", size:" + actualLabel.length);
    for (int i = 0; i < txxx.length; i++) {
      System.out.print(txxx[i] + ", ");
      System.out.print(actualLabel[i]);
      System.out.println();
    }

    TestCase.assertTrue(Arrays.equals(actualLabel, txxx));

    table.close();
    table1.close();
  }

  @Test
  public void testBooster() throws XGBoostError {
    Schema schema = Schema.builder()
      .column(DType.FLOAT32, "A")
      .column(DType.FLOAT32, "B")
      .column(DType.FLOAT32, "C")
      .column(DType.FLOAT32, "D")
      .column(DType.FLOAT32, "label")
      .build();

    int maxBin = 16;
    int round = 100;
    //set params
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 2);
        put("objective", "multi:softprob");
        put("num_class", 3);
        put("num_round", round);
        put("num_workers", 1);
        put("tree_method", "gpu_hist");
        put("predictor", "gpu_predictor");
        put("max_bin", maxBin);
      }
    };

    Table tmpTable = Table.readCSV(schema, new File("./src/test/resources/iris.data.csv"));
    System.out.println(tmpTable.getRowCount());

    CudfTable cudfTable = new CudfTable(tmpTable, new int[]{0, 1, 2, 3}, new int[]{4}, null, null);

    //set watchList
    HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();

    DMatrix dMatrix1 = new DMatrix(cudfTable.getFeatureArrayInterface(), Float.NaN, 1);
    dMatrix1.setLabel(cudfTable.getLabelArrayInterface());
    watches.put("train", dMatrix1);
    Booster model1 = XGBoost.train(dMatrix1, paramMap, 100, watches, null, null);

    java.util.List<CudfTable> rapidsTables = new java.util.LinkedList<CudfTable>();
    rapidsTables.add(cudfTable);
    DMatrix incrementalDMatrix = new DMatrix(rapidsTables.iterator(), Float.NaN, maxBin, 1);
    //set watchList
    HashMap<String, DMatrix> watches1 = new HashMap<String, DMatrix>();
    watches1.put("train", incrementalDMatrix);
    Booster model2 = XGBoost.train(incrementalDMatrix, paramMap, 100, watches1, null, null);

    float[][] predicat1 = model1.predict(dMatrix1);
    float[][] predicat2 = model2.predict(dMatrix1);

    for (int i = 0; i < tmpTable.getRowCount(); i++) {
      TestCase.assertTrue(predicat1[i][0] - predicat2[i][0] < 0.000000001);
    }

    tmpTable.close();
  }

  private float[] convertFloatTofloat(Float[] in) {
    return Floats.toArray(Arrays.asList(in));
  }
}
