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

package ml.dmlc.xgboost4j.scala.spark.rapids

import scala.collection.JavaConverters._
import ai.rapids.cudf.Table
import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.java.spark.rapids.{GpuColumnBatch, GpuColumnVector}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import org.apache.commons.logging.LogFactory
import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch

import scala.collection.Iterator

/**
 * Convert Cudf table to Rows
 *
 */
private[spark] object GpuTransform {
  private val logger = LogFactory.getLog("GpuTransform")

  def buildRddIterator(colNames: Seq[String],
      colDataItrs: Seq[Iterator[Row]],
      rawIter: Iterator[Row]): Iterator[Row] = {

    require(colNames.length == colDataItrs.length)
    colNames.zip(colDataItrs).foldLeft(rawIter) {
      case (outIt, (cName, dataIt)) =>
        if (cName.nonEmpty && dataIt.nonEmpty) {
          outIt.zip(dataIt).map {
            case (origRow, dataRow) => Row.fromSeq(origRow.toSeq ++ dataRow.toSeq)
          }
        } else outIt
    }
  }

  // get column name, null | undefined will be casted to ""
  private[this] def getColumnName(params: Params)(param: Param[String]): String = {
    if (params.isDefined(param)) {
      val colName = params.getOrDefault(param)
      if (colName != null) colName else ""
    } else ""
  }

  def getColumnNames(params: Params)(cols: Param[String]*): Seq[String] = {
    val getName = getColumnName(params)(_)
    cols.map(getName)
  }

  def cudfTableToRowIterator(iter: Iterator[Table],
      schema: StructType,
      booster: Broadcast[Booster],
      isLocal: Boolean,
      missingValue: Float,
      featureIndices: Seq[Int],
      predictFunc: (Broadcast[Booster], DMatrix, Iterator[Row]) => Iterator[Row],
      sampler: Option[GpuSampler] = None): Iterator[Row] = {

    // UnsafeProjection is not serializable so do it on the executor side
    val toUnsafe = UnsafeProjection.create(schema)
    new Iterator[Row] {
      private var batchCnt = 0
      private var converter: RowConverter = null

      // GPU batches read in must be closed by the receiver (us)
      @transient var cb: ColumnarBatch = null
      var it: Iterator[Row] = null

      TaskContext.get().addTaskCompletionListener[Unit](_ => {
        if (batchCnt > 0) {
          Rabit.shutdown()
        }
        closeCurrentBatch()
      })

      private def closeCurrentBatch(): Unit = {
        if (cb != null) {
          cb.close()
          cb = null
        }
      }

      def loadNextBatch(): Unit = {
        closeCurrentBatch()
        if (it != null) {
          it = null
        }
        if (iter.hasNext) {
          val table = iter.next()

          if (batchCnt == 0) {
            // do we really need to involve rabit in transform?
            // Init rabit
            val rabitEnv = Map(
              "DMLC_TASK_ID" -> TaskContext.getPartitionId().toString)
            Rabit.init(rabitEnv.asJava)

            converter = new RowConverter(schema,
              (0 until table.getNumberOfColumns).map(table.getColumn(_).getType))
          }

          val devCb = GpuColumnVector.from(table)

          try {
            cb = new ColumnarBatch(
              GpuColumnVector.extractColumns(devCb).map(_.copyToHost()),
              devCb.numRows())

            val rowIterator = cb.rowIterator().asScala
              .map(toUnsafe)
              .map(converter.toExternalRow(_))

            // Create DMatrix

            var dm: DMatrix = null

            val gpuId = GpuDeviceManager.getGpuId(isLocal)

            val columnBatch = new GpuColumnBatch(table, schema, sampler.getOrElse(null))
            dm = new DMatrix(gpuId, missingValue,
                columnBatch.getAsColumnData(featureIndices: _*): _*)

            it = {
              if (dm == null) {
                Iterator.empty
              } else {
                try {
                  // set some params of gpu related to booster
                  // - gpu id
                  // - predictor: Force to gpu predictor since native doesn't save predictor.

                  booster.value.setParam("gpu_id", gpuId.toString)
                  booster.value.setParam("predictor", "gpu_predictor")
                  logger.info("XGBoost transform GPU pipeline using device: " + gpuId)

                  predictFunc(booster, dm, rowIterator)
                } finally {
                  dm.delete()
                }
              }
            }
          } finally {
            batchCnt += 1
            devCb.close()
            table.close()
          }
        }
      }

      override def hasNext: Boolean = {
        val itHasNext = it != null && it.hasNext
        if (!itHasNext) {
          loadNextBatch()
          it != null && it.hasNext
        } else {
          itHasNext
        }
      }

      override def next(): Row = {
        if (it == null || !it.hasNext) {
          loadNextBatch()
        }
        if (it == null) {
          throw new NoSuchElementException()
        }
        it.next()
      }
    }
  }
}
