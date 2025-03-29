/**
 * @File:   footprint_extruder.cpp
 * @Author: Haozhe Xie
 * @Date:   2024-02-12 13:07:49
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-03-23 18:49:41
 * @Email:  root@haozhexie.com
 *
 * References:
 * https://github.com/hzxie/RMNet/blob/master/extensions/flow_affine_transformation/flow_affine_transformation.cpp
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define N_DIM 5

#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <Python.h>
#include <numpy/arrayobject.h>

// Initialize Numpy and make sure that be called only once.
// Without initialize, both PyArray_SimpleNew and PyArray_Return cause
// segmentation fault error.
int initNumpy() {
  import_array();
  return 1;
}
const static int NUMPY_INITIALIZED = initNumpy();

// NOTE: The second dummy parameters is for overloading
inline short getValueFromPyObject(PyObject *obj, const short *_) {
  return static_cast<short>(PyLong_AsLong(obj));
}

// NOTE: The second dummy parameters is for overloading
inline std::string getValueFromPyObject(PyObject *obj, const std::string *_) {
  char *c_str = PyBytes_AsString(PyUnicode_AsUTF8String(obj));
  std::string str(c_str);
  // The following statements cause segmentation fault.
  // Py_DECREF(obj);
  // PyMem_Free(c_str);
  return str;
}

template <typename kT, typename vT>
std::map<kT, vT> getMapFromPyObject(PyObject *obj, const kT &kP, const vT &vP) {
  std::map<kT, vT> map;
  Py_ssize_t idx = 0;
  PyObject *key;
  PyObject *value;
  while (PyDict_Next(obj, &idx, &key, &value)) {
    kT k = getValueFromPyObject(key, &kP);
    vT v = getValueFromPyObject(value, &vP);
    map[k] = v;
  }
  return map;
}

PyObject *
getNumpyArrayFromVector(const std::vector<std::vector<short>> &vec2d) {
  // Determine the dimensions of the 2D std::vector
  npy_intp dims[2] = {static_cast<npy_intp>(vec2d.size()),
                      static_cast<npy_intp>(vec2d[0].size())};
  // Convert the 2D std::vector to a NumPy array
  PyObject *numpy_array = PyArray_SimpleNew(2, dims, NPY_UINT16);
  short *data = reinterpret_cast<short *>(
      PyArray_DATA(reinterpret_cast<PyArrayObject *>(numpy_array)));
  // Copy the data from the 2D std::vector to the NumPy array using pointer
  // arithmetic
  for (const auto &row : vec2d) {
    std::copy(row.begin(), row.end(), data);
    data += row.size();
  }
  // Return the NumPy array
  return numpy_array;
}

inline size_t getArrayIndex(short rowIdx, short colIdx, short nCols) {
  return rowIdx * nCols + colIdx;
}

inline short getSemanticID(short instanceID,
                           const std::map<std::string, short> &segInsMap) {
  if (instanceID < segInsMap.at("BLDG_INS_MIN_ID")) {
    return instanceID;
  } else if (instanceID >= segInsMap.at("CAR_INS_MIN_ID")) {
    return segInsMap.at("CAR_SEMANTIC_ID");
  } else {
    // WARN: The semantic labels for buildings would be merged to facade.
    return segInsMap.at("BLDG_FACADE_SEMANTIC_ID");
  }
}

inline bool isNeighboringValueSame(const short *map, short x, short y,
                                   short width, short scale) {
  // Unroll the for-loop for faster speed
  // For the next point, the index is x/y + scale instead of x/y + 1.
  short c_value = map[getArrayIndex(y, x, width)];
  std::array<short, 8> nbr_values{
      map[getArrayIndex(y - scale, x - scale, width)],
      map[getArrayIndex(y - scale, x, width)],
      map[getArrayIndex(y - scale, x + scale, width)],
      map[getArrayIndex(y, x - scale, width)],
      map[getArrayIndex(y, x + scale, width)],
      map[getArrayIndex(y + scale, x - scale, width)],
      map[getArrayIndex(y + scale, x, width)],
      map[getArrayIndex(y + scale, x + scale, width)],
  };
  for (size_t i = 0; i < nbr_values.size(); ++i) {
    if (c_value != nbr_values[i]) {
      return false;
    }
  }
  return true;
}

inline bool isBorder(short x, short y, short z, short height, short width,
                     short scale, bool incBtmPts, const short *segMap,
                     const short *tpDwnHgtFld, const short *btmUpHgtFld) {
  int idx = getArrayIndex(y, x, width);
  if (z > tpDwnHgtFld[idx] - scale || (z == btmUpHgtFld[idx] && incBtmPts)) {
    return true;
  }
  if (x < scale || x >= width - scale - 1 || y < scale || y >= height - scale - 1) {
    return true;
  }
  return !isNeighboringValueSame(segMap, x, y, width, scale) ||
         !isNeighboringValueSame(tpDwnHgtFld, x, y, width, scale);
}

static PyObject *getPointsFromProjection(PyObject *self, PyObject *args) {
  PyObject *pyIncBtmPts;
  PyObject *pyClasses;
  PyObject *pyScales;
  PyObject *pySegInsMap;
  PyArrayObject *pySegMap;
  PyArrayObject *pyTpDwnHgtFld;
  PyArrayObject *pyBtmUpHgtFld;
  PyArrayObject *pyPtsMap;
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!", &PyBool_Type, &pyIncBtmPts,
                        &PyDict_Type, &pyClasses, &PyDict_Type, &pyScales,
                        &PyDict_Type, &pySegInsMap, &PyArray_Type, &pySegMap,
                        &PyArray_Type, &pyTpDwnHgtFld, &PyArray_Type,
                        &pyBtmUpHgtFld, &PyArray_Type, &pyPtsMap)) {
    return NULL;
  }

  // NOTE: The last two dummy parameters are used for overloading
  bool incBtmPts = PyObject_IsTrue(pyIncBtmPts);
  auto classes =
      getMapFromPyObject<short, std::string>(pyClasses, 0, std::string());
  auto scales =
      getMapFromPyObject<std::string, short>(pyScales, std::string(), 0);
  auto segInsMap =
      getMapFromPyObject<std::string, short>(pySegInsMap, std::string(), 0);

  npy_intp *mapSize = PyArray_SHAPE(pyPtsMap);
  short height = mapSize[0], width = mapSize[1];
  assert(height <= std::numeric_limits<std::short>::max());
  assert(width <= std::numeric_limits<std::short>::max());
  assert(height * width <= std::numeric_limits<std::size_t>::max());

  short *segMap = static_cast<short *>(PyArray_DATA(pySegMap));
  short *tpDwnHgtFld = static_cast<short *>(PyArray_DATA(pyTpDwnHgtFld));
  short *btmUpHgtFld = static_cast<short *>(PyArray_DATA(pyBtmUpHgtFld));
  bool *ptsMap = static_cast<bool *>(PyArray_DATA(pyPtsMap));

  std::vector<std::vector<short>> points;
  for (short i = 0; i < height; ++i) {
    for (short j = 0; j < width; ++j) {
      size_t idx = getArrayIndex(i, j, width);
      // The pixels in segMap, tpDwnHgtFld, and btmUpHgtFld are densified.
      // The original points (without densification) are defined in ptsMap.
      if (!ptsMap[idx]) {
        continue;
      }

      short instanceID = segMap[idx];
      short semanticID = getSemanticID(instanceID, segInsMap);
      short scale = scales[classes[semanticID]];

      for (short k = btmUpHgtFld[idx]; k <= tpDwnHgtFld[idx]; k += scale) {
        // Make all objects hallow
        if (!isBorder(j, i, k, height, width, scale, incBtmPts, segMap,
                      tpDwnHgtFld, btmUpHgtFld)) {
          continue;
        }
        // Building Roof Handler (Recover roof instance ID)
        if (k > tpDwnHgtFld[idx] - scale &&
            semanticID == segInsMap.at("BLDG_FACADE_SEMANTIC_ID")) {
          instanceID += segInsMap.at("ROOF_INS_OFFSET");
        }
        points.push_back({j, i, k, scale, instanceID});
      }
    }
  }
  // Returning empty array causes segmentation fault.
  if (points.size() != 0) {
    return getNumpyArrayFromVector(points);
  }
  Py_RETURN_NONE;
}

static PyMethodDef extensionMethods[] = {
    {"get_points_from_projection", getPointsFromProjection, METH_VARARGS,
     "Generate the initial points from multiple projection maps"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT, "footprint_extruder",
    "Generate the initial points from multiple projection maps", -1,
    extensionMethods};

PyMODINIT_FUNC PyInit_footprint_extruder(void) {
  return PyModule_Create(&cModPyDem);
}