package ml.dmlc.xgboost4j.java.rapids;

import java.util.Iterator;

public class ColumnBatch implements AutoCloseable {
  private String cudfJson;
  private CudfTable table;
  public ColumnBatch(CudfTable table, String cudfJson) {
    this.table = table;
    this.cudfJson = cudfJson;
  }

  // called from native
  public String getCudfJsonStr() {
    return cudfJson;
  }

  // called from native
  @Override
  public void close() throws Exception {
    table.close();
  }

  public static class ColumnBatchBatchIterator implements Iterator<ColumnBatch> {
    private Iterator<CudfTable> base;

    public ColumnBatchBatchIterator(Iterator<CudfTable> base) {
      this.base = base;
    }

    @Override
    public boolean hasNext() {
      return base.hasNext();
    }

    @Override
    public ColumnBatch next() {
      CudfTable table = base.next();
      return new ColumnBatch(table, table.getCudfJsonStr());
    }
  }
}
