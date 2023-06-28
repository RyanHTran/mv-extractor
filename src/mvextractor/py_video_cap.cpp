#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <opencv2/core/core.hpp>

#include "video_cap.hpp"
#include "mat_to_ndarray.hpp"

typedef struct {
    PyObject_HEAD
    VideoCap vcap;
} VideoCapObject;


static void
VideoCap_dealloc(VideoCapObject *self)
{
    self->vcap.release();
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *
VideoCap_open(VideoCapObject *self, PyObject *args)
{
    const char *url;
    int frame_type;
    int iframe_width;
    int iframe_height;
    int mv_res_reduction;

    if (!PyArg_ParseTuple(args, "sCiii", &url, &frame_type, &iframe_width, &iframe_height, &mv_res_reduction))
        Py_RETURN_FALSE;

    if (!self->vcap.open(url, (char)frame_type,  iframe_width, iframe_height, mv_res_reduction))
        Py_RETURN_FALSE;

    Py_RETURN_TRUE;
}

static PyObject *
VideoCap_read(VideoCapObject *self, PyObject *Py_UNUSED(ignored))
{
    PyArrayObject *frame = NULL;
    int width = 0;
    int height = 0;
    int step = 0;
    int cn = 0;
    int gop_idx = -1;
    int gop_pos = 0;

    char frame_type[2] = "?";

    PyObject *ret = Py_True;

    if (!self->vcap.read(&frame, &step, &width, &height, &cn, frame_type, &gop_idx, &gop_pos)) {
        width = 0;
        height = 0;
        step = 0;
        cn = 0;
        frame = (PyArrayObject *)Py_None;
        ret = Py_False;
    }

    return Py_BuildValue("(ONsii)", ret, frame, (const char*)frame_type, gop_idx, gop_pos);
}

static PyObject *
VideoCap_read_accumulate(VideoCapObject *self, PyObject *Py_UNUSED(ignored))
{
    uint8_t *frame = NULL;
    int width = 0;
    int height = 0;
    int step = 0;
    int cn = 0;
    int gop_idx = -1;
    int gop_pos = 0;

    PyArrayObject *accumulated_mv = NULL;
    MVS_DTYPE num_mvs = 0;
    char frame_type[2] = "?";

    PyObject *ret = Py_True;
    
    if (!self->vcap.read_accumulate(&frame, &step, &width, &height, &cn, frame_type, &accumulated_mv, &num_mvs, &gop_idx, &gop_pos)) {
        width = 0;
        height = 0;
        step = 0;
        cn = 0;
        accumulated_mv = (PyArrayObject *)Py_None;
        ret = Py_False;
    }

    npy_intp dims[3] = {height, width, cn};
    PyObject* frame_nd = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, frame);

    return Py_BuildValue("(ONOsii)", ret, frame_nd, accumulated_mv, (const char*)frame_type, gop_idx, gop_pos);
}


static PyObject *
VideoCap_release(VideoCapObject *self, PyObject *Py_UNUSED(ignored))
{
    self->vcap.release();
    Py_RETURN_NONE;
}


static PyMethodDef VideoCap_methods[] = {
    {"open", (PyCFunction) VideoCap_open, METH_VARARGS, "Open a video file or device with given filename/url"},
    {"read", (PyCFunction) VideoCap_read, METH_NOARGS, "Grab and decode the next frame and motion vectors"},
    {"read_accumulate", (PyCFunction) VideoCap_read_accumulate, METH_NOARGS, "Decode the grabbed frame and accumulate the motion vectors"},
    {"release", (PyCFunction) VideoCap_release, METH_NOARGS, "Release the video device and free ressources"},
    {NULL}  /* Sentinel */
};


static PyTypeObject VideoCapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "videocap.VideoCap",
    .tp_basicsize = sizeof(VideoCapObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) VideoCap_dealloc,
    .tp_vectorcall_offset = NULL,
    .tp_getattr = NULL,
    .tp_setattr = NULL,
    .tp_as_async = NULL,
    .tp_repr = NULL,
    .tp_as_number = NULL,
    .tp_as_sequence = NULL,
    .tp_as_mapping = NULL,
    .tp_hash = NULL,
    .tp_call = NULL,
    .tp_str = NULL,
    .tp_getattro = NULL,
    .tp_setattro = NULL,
    .tp_as_buffer = NULL,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Video Capture Object",
    .tp_traverse = NULL,
    .tp_clear = NULL,
    .tp_richcompare = NULL,
    .tp_weaklistoffset = 0,
    .tp_iter = NULL,
    .tp_iternext = NULL,
    .tp_methods = VideoCap_methods,
    .tp_members = NULL,
    .tp_getset = NULL,
    .tp_base = NULL,
    .tp_dict = NULL,
    .tp_descr_get = NULL,
    .tp_descr_set = NULL,
    .tp_dictoffset = 0,
    .tp_init = NULL,
    .tp_alloc = NULL,
    .tp_new = PyType_GenericNew,
    .tp_free = NULL,
    .tp_is_gc = NULL,
    .tp_bases = NULL,
    .tp_mro = NULL,
    .tp_cache = NULL,
    .tp_subclasses = NULL,
    .tp_weaklist = NULL,
    .tp_del = NULL,
    .tp_version_tag = 0,
    .tp_finalize  = NULL,
};


static PyModuleDef videocapmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "videocap",
    .m_doc = "Capture video frames and motion vectors from a H264 encoded stream.",
    .m_size = -1,
};


PyMODINIT_FUNC
PyInit_videocap(void)
{
    Py_Initialize();  // maybe not needed
    import_array();

    PyObject *m;
    if (PyType_Ready(&VideoCapType) < 0)
        return NULL;

    m = PyModule_Create(&videocapmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&VideoCapType);
    PyModule_AddObject(m, "VideoCap", (PyObject *) &VideoCapType);
    return m;
}
