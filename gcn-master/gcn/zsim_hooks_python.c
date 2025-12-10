#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* 使用相对路径引入 zsim hooks，避免绝对路径导致的 includePath 报错 */
#include "../../../ramulator-pim-master/zsim-ramulator/misc/hooks/zsim_hooks.h"

static PyObject *py_zsim_roi_begin(PyObject *self, PyObject *args)
{
    zsim_roi_begin();
    Py_RETURN_NONE;
}

static PyObject *py_zsim_roi_end(PyObject *self, PyObject *args)
{
    zsim_roi_end();
    Py_RETURN_NONE;
}

static PyObject *py_zsim_PIM_function_begin(PyObject *self, PyObject *args)
{
    zsim_PIM_function_begin();
    Py_RETURN_NONE;
}

static PyObject *py_zsim_PIM_function_end(PyObject *self, PyObject *args)
{
    zsim_PIM_function_end();
    Py_RETURN_NONE;
}

static PyMethodDef ZSimMethods[] = {
    {"zsim_roi_begin", py_zsim_roi_begin, METH_NOARGS, "Mark ROI begin"},
    {"zsim_roi_end", py_zsim_roi_end, METH_NOARGS, "Mark ROI end"},
    {"zsim_PIM_function_begin", py_zsim_PIM_function_begin, METH_NOARGS, "Mark PIM begin"},
    {"zsim_PIM_function_end", py_zsim_PIM_function_end, METH_NOARGS, "Mark PIM end"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef zsimmodule = {
    PyModuleDef_HEAD_INIT,
    "zsim_hooks_python",
    NULL,
    -1,
    ZSimMethods};

PyMODINIT_FUNC PyInit_zsim_hooks_python(void)
{
    return PyModule_Create(&zsimmodule);
}