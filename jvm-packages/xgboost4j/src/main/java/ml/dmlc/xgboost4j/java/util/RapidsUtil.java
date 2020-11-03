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

package ml.dmlc.xgboost4j.java.util;

import java.util.ArrayList;

import org.codehaus.jettison.json.JSONException;
import org.codehaus.jettison.json.JSONObject;

public class RapidsUtil {
  public static JSONObject getArrayInterfaceJsonObject(long ptr, long shape, String typeStr) {
    JSONObject jsonObject = new JSONObject();
    try {
      ArrayList<Long> shapeList = new ArrayList<>();
      shapeList.add(shape);
      jsonObject.put("shape", shapeList);
      ArrayList<Object> dataList = new ArrayList<>();
      dataList.add(ptr);
      dataList.add(false);
      jsonObject.put("data", dataList);
      jsonObject.put("typestr", typeStr);
      jsonObject.put("version", 1);
    } catch (JSONException e) {
      e.printStackTrace();
      return null;
    }
    return jsonObject;
  }
}
