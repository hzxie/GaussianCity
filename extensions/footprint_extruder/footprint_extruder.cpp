/**
 * @File:   footprint_extruder.cpp
 * @Author: Haozhe Xie
 * @Date:   2024-02-12 13:07:49
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-02-12 19:05:29
 * @Email:  root@haozhexie.com
 *
 * References:
 * https://github.com/hzxie/RMNet/blob/master/extensions/flow_affine_transformation/flow_affine_transformation.cpp
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define N_DIM 5
#define CAR_SEMANTIC_ID 3
#define BLDG_FACADE_SEMANTIC_ID 7
#define BLDG_ROOF_SEMANTIC_ID 8
#define BLDG_INS_MIN_ID 100
#define ROOF_INS_OFFSET 1
#define CAR_INS_MIN_ID 5000

#include <iostream>

#include <cassert>
#include <cmath>
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

/* NOTE: The second dummy parameters is for overloading */
inline int16_t getValueFromPyObject(PyObject *obj, const int16_t *_) {
  return static_cast<int16_t>(PyLong_AsLong(obj));
}

/* NOTE: The second dummy parameters is for overloading */
inline std::string getValueFromPyObject(PyObject *obj, const std::string *_) {
  char *c_str = PyBytes_AsString(PyUnicode_AsUTF8String(obj));
  std::string str(c_str);
  Py_DECREF(obj);
  // The following statement causes segmentation fault.
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

PyObject *getNumpyArrayFromVector(std::vector<std::vector<int16_t>> vec2d) {
  // Determine the dimensions of the 2D std::vector
  npy_intp dims[2] = {static_cast<npy_intp>(vec2d.size()),
                      static_cast<npy_intp>(vec2d[0].size())};
  // Convert the 2D std::vector to a NumPy array
  PyObject *numpy_array = PyArray_SimpleNew(2, dims, NPY_UINT16);
  int16_t *data = reinterpret_cast<int16_t *>(
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

inline size_t getArrayIndex(int16_t rowIdx, int16_t colIdx, int16_t nCols) {
  return rowIdx * nCols + colIdx;
}

inline int16_t getSemanticID(int16_t instanceID) {
  if (instanceID < BLDG_INS_MIN_ID) {
    return instanceID;
  } else if (instanceID >= CAR_INS_MIN_ID) {
    return CAR_SEMANTIC_ID;
  } else {
    // WARN: The semantic labels for buildings would be merged to facade.
    return BLDG_FACADE_SEMANTIC_ID;
  }
}

inline bool isBorder(int16_t x, int16_t y, int16_t z, int16_t height,
                     int16_t width, const int16_t *segMap,
                     const int16_t *tpDwnHgtFld, const int16_t *btmUpHgtFld) {
  int idx = getArrayIndex(y, x, width);
  if (z == btmUpHgtFld[idx] || z == tpDwnHgtFld[idx]) {
    return true;
  }
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
    return true;
  }

  // The coordinates of neighboring points
  int16_t neg_x = x - 1, pos_x = x + 1;
  int16_t neg_y = y - 1, pos_y = y + 1;
  // Unroll the for-loop for faster speed
  std::array<int16_t, 9> values{
      segMap[getArrayIndex(neg_y, neg_x, width)],
      segMap[getArrayIndex(neg_y, x, width)],
      segMap[getArrayIndex(neg_y, pos_x, width)],
      segMap[getArrayIndex(y, neg_x, width)],
      segMap[idx],
      segMap[getArrayIndex(y, pos_x, width)],
      segMap[getArrayIndex(pos_y, neg_x, width)],
      segMap[getArrayIndex(pos_y, x, width)],
      segMap[getArrayIndex(pos_y, pos_x, width)],
  };

  for (size_t i = 1; i < values.size(); ++ i) {
    if (values[i - 1] != values[i]) {
      return true;
    }
  }
  return false;
}

static PyObject *getPointsFromProjection(PyObject *self, PyObject *args) {
  PyObject *pyClasses;
  PyObject *pyScales;
  PyArrayObject *pySegMap;
  PyArrayObject *pyTpDwnHgtFld;
  PyArrayObject *pyBtmUpHgtFld;
  PyArrayObject *pyPtsMap;
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", &PyDict_Type, &pyClasses,
                        &PyDict_Type, &pyScales, &PyArray_Type, &pySegMap,
                        &PyArray_Type, &pyTpDwnHgtFld, &PyArray_Type,
                        &pyBtmUpHgtFld, &PyArray_Type, &pyPtsMap)) {
    return NULL;
  }

  auto classes =
      getMapFromPyObject<int16_t, std::string>(pyClasses, 0, std::string());
  auto scales =
      getMapFromPyObject<std::string, int16_t>(pyScales, std::string(), 0);

  npy_intp *mapSize = PyArray_SHAPE(pyPtsMap);
  int16_t height = mapSize[0], width = mapSize[1];
  assert(height <= std::numeric_limits<std::uint16_t>::max());
  assert(width <= std::numeric_limits<std::uint16_t>::max());
  assert(height * width <= std::numeric_limits<std::size_t>::max());

  int16_t *segMap = static_cast<int16_t *>(PyArray_DATA(pySegMap));
  int16_t *tpDwnHgtFld = static_cast<int16_t *>(PyArray_DATA(pyTpDwnHgtFld));
  int16_t *btmUpHgtFld = static_cast<int16_t *>(PyArray_DATA(pyBtmUpHgtFld));
  bool *ptsMap = static_cast<bool *>(PyArray_DATA(pyPtsMap));

  std::vector<std::vector<int16_t>> points;
  for (int16_t i = 0; i < height; ++i) {
    for (int16_t j = 0; j < width; ++j) {
      size_t idx = getArrayIndex(i, j, width);
      // The pixels in segMap, tpDwnHgtFld, and btmUpHgtFld are densified.
      // The original points (without densification) are defined in ptsMap.
      if (!ptsMap[idx]) {
        continue;
      }
      // The semantic labels for buildings would be merged to facade in
      // getSemanticID().
      int16_t instanceID = segMap[idx];
      int16_t semanticID = getSemanticID(instanceID);
      int16_t scale = scales[classes[semanticID]];
      for (int16_t k = btmUpHgtFld[idx]; k <= tpDwnHgtFld[idx]; k += scale) {
        // Make all objects hallow
        if (!isBorder(j, i, k, height, width, segMap, tpDwnHgtFld,
                      btmUpHgtFld)) {
          continue;
        }
        // Building Roof Handler (Recover roof instance ID)
        if (k == tpDwnHgtFld[idx] && semanticID == BLDG_FACADE_SEMANTIC_ID) {
          instanceID += ROOF_INS_OFFSET;
        }
        points.push_back({j, i, k, scale, instanceID});
      }
    }
  }
  return getNumpyArrayFromVector(points);
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