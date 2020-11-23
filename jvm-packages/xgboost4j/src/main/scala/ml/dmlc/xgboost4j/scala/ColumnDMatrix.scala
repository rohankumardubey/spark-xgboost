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

package ml.dmlc.xgboost4j.scala

import ml.dmlc.xgboost4j.java.rapids.CudfTable

import _root_.scala.collection.JavaConverters._
import ml.dmlc.xgboost4j.java.{XGBoostError, DMatrix => JDMatrix}

class ColumnDMatrix private[scala](private[scala] override val jDMatrix: JDMatrix)
    extends DMatrix(jDMatrix) {

  /**
   * Create DMatrix from column array interface
   * @param dataJson array interface
   * @param missing missing value
   * @param nthread threads number
   */
  @throws(classOf[XGBoostError])
  def this(dataJson: String, missing: Float, nthread: Int) {
    this(new JDMatrix(dataJson, missing, nthread))
  }

  /**
   * Create DeviceDMatrix incrementally from iterator of column array iterface
   * @param rapidsTableItr
   * @param missing
   * @param maxBin
   * @param nthread
   */
  def this(rapidsTableItr: Iterator[CudfTable], missing: Float, maxBin: Int, nthread: Int) {
    this(new JDMatrix(rapidsTableItr.asJava, missing, maxBin, nthread))
  }

  /**
   * set label of dmatrix from column array interface
   *
   * @param labelJson label column array interface
   */
  @throws(classOf[XGBoostError])
  def setLabel(labelJson: String): Unit = {
    jDMatrix.setLabel(labelJson)
  }

  /**
   * set label of dmatrix from column array interface
   *
   * @param weightJson weight column array interface
   */
  @throws(classOf[XGBoostError])
  def setWeight(weightJson: String): Unit = {
    jDMatrix.setWeight(weightJson)
  }

  /**
   * set label of dmatrix from column array interface
   *
   * @param baseMarginJson base margin column array interface
   */
  @throws(classOf[XGBoostError])
  def setBaseMargin(baseMarginJson: String): Unit = {
    jDMatrix.setBaseMargin(baseMarginJson)
  }

}
