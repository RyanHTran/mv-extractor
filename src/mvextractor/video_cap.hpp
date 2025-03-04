#include <thread>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <math.h>

// FFMPEG
extern "C" {
#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include "time_cvt.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL accumulator_ARRAY_API
#include <numpy/arrayobject.h>

// for changing the dtype of motion vector
#define MVS_DTYPE int32_t
#define MVS_DTYPE_NP NPY_INT32
#define MV_ELEMS 7

// whether or not to print some debug info
//#define DEBUG


struct Image_FFMPEG
{
    unsigned char* data;
    int width;
    int height;
    int step;
    int cn;
};


/**
* Decode frames and motion vectors from a H264 encoded video file or RTSP stream.
*
* Implements a VideoCap object similar to OpenCV's VideoCapture. For details
* see (https://docs.opencv.org/4.1.0/d8/dfe/classcv_1_1VideoCapture.html).
* The class is intended to open a H264 encoded video file or RTSP stream by
* providing the according file path or stream url to the `open` method.
* Upon sucessful opening of the stream, the `read` method can be called in
* a loop each time yielding the next decoded frame of the stream as well as
* frame side data, such as motion vectors (as specified per H264 standard).
* Instead of calling read, the two methods `grab` and `retrieve` can be used.
* `grab` performs reading of the next frame from the stream and decoding which
* is fast. `retrieve` performs color space conversion of the frame and motion
* vector extraction which is slower. Splitting up `read` like this allows to
* generate timestamps which are close to another in case multi-camera setups
* are used and captured frames should be close to another.
*
*/
class VideoCap {

private:
    const char *url;
    AVDictionary *opts;
    AVCodec *codec;
    AVFormatContext *fmt_ctx;
    AVCodecContext *video_dec_ctx;
    AVStream *video_stream;
    int video_stream_idx;
    AVPacket packet;
    AVFrame *frame;
    AVFrame **out_frames;
    struct SwsContext *img_convert_ctx;
    int64_t frame_number;
    bool is_rtsp;
    PyArrayObject **out_mvs;
    PyArrayObject **out_mvs_backward;
    int *prev_locations;
    int *curr_locations;
    int gop_idx;
    int gop_pos;
    char frame_type;
    int mv_res_reduction;
    int iframe_width;
    int iframe_height;
    int gop_size;
    bool finished_reading;
#if USE_AV_INTERRUPT_CALLBACK
    AVInterruptCallbackMetadata interrupt_metadata;
#endif

    /** Determines whether the input is a video file or an RTSP stream
    *
    * @param format_names A comma separated list of formats which correspond to
    *     to the the input. This list is stored in the `iformat->name` field of
    *     the stream's AVFormatContext.
    *
    * @retval true if the format names contain "rtsp" which means the input url
    *     correpsonds to an RTSP stream, false if the input is a video file.
    */
    bool check_format_rtsp(const char *format_names);

    void reset_accumulate(bool backwards);

    /** Reads the next video frame and motion vectors from the stream
    *
    * @retval true if a new video frame could be read and decoded, false
    *    otherwise (e.g. at the end of the stream).
    */
    bool grab(char *frame_type);

    /** Decodes and returns the grabbed frame and motion vectors
    *
    * @param frame Pointer to the raw data of the decoded video frame. The
    *    frame is stored as a C contiguous array of shape (height, width, 3) and
    *    can be converted into a cv::Mat by using the constructor
    *    `cv::Mat cv_frame(height, width, CV_MAKETYPE(CV_8U, 3), frame)`.
    *    Note: A subsequent call of `retrieve` will reuse the same memory for
    *          storing the new frame. If you want a frame to persist for a longer
    *          perdiod of time, allocate a new array and memcopy the raw frame
    *          data into it. After usage you have to manually free this copied
    *          array.
    *
    * @param width Width of the returned frame in pixels.
    *
    * @param height Height of the returned frame in pixels.
    *
    * @param frame_type Either "P", "B" or "I" indicating whether it is an
    *    intra-coded frame (I), a predicted frame with only references to past
    *    frames (P) or reference to both past and future frames (B). Motion
    *    vectors are only returned for "P" and "B" frames.
    *
    * @retval true if the grabbed video frame and motion vectors could be
    *    decoded and returned successfully, false otherwise.
    */
    bool retrieve(AVFrame *out_frame, int *step, int *width, int *height, int *cn, int *gop_idx, int *gop_pos);

    // Performs the same decoding as `retrieve` and also accumulates motion vectors into 2D array
    void accumulate(AVFrame *out_frame, PyArrayObject *out_mv, bool backwards);

public:

    /** Constructor */
    VideoCap();

    /** Destroy the VideoCap object and free all ressources */
    void release(void);

    /** Open a video file or RTSP url
    *
    * The stream must be H264 encoded. Otherwise, undefined behaviour is
    * likely.
    *
    * @param url Relative or fully specified file path or an RTSP url specifying
    *     the location of the video stream. Example "vid.flv" for a video
    *     file located in the same directory as the source files. Or
    *     "rtsp://xxx.xxx.xxx.xxx:554" for an IP camera streaming via RTSP.
    *
    * @retval true if video file or url could be opened sucessfully, false
    *     otherwise.
    */
    bool open(const char *url, char frame_type, int iframe_width, int iframe_height, int mv_res_reduction, int gop_size);
    
    /** Convenience wrapper which combines a call of `grab` and `retrieve`.
    *
    *   The parameters and return value correspond to the `retrieve` method.
    */
    bool read(PyArrayObject **frame, int *step, int *width, int *height, int *cn, char *frame_type, int *gop_idx, int *gop_pos);

    bool read_gop(PyObject **frames, int *step, int *width, int *height, int *cn, char *frame_type, int *gop_idx);

    bool read_accumulate(PyArrayObject **frame, int *step, int *width, int *height, int *cn, char *frame_type, PyArrayObject **accumulated_mv, int *gop_idx, int *gop_pos);

    bool read_accumulate_gop(PyObject **frames, int *step, int *width, int *height, int *cn, char *frame_type, PyObject **forward_mvs, PyObject **backward_mvs, int *gop_idx);
};
