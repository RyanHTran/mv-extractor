#define NO_IMPORT_ARRAY

#include "video_cap.hpp"
#include <vector>

VideoCap::VideoCap() {
    this->opts = NULL;
    this->codec = NULL;
    this->fmt_ctx = NULL;
    this->video_dec_ctx = NULL;
    this->video_stream = NULL;
    this->video_stream_idx = -1;
    this->frame = NULL;
    this->out_frames = NULL;
    this->img_convert_ctx = NULL;
    this->frame_number = 0;
    this->is_rtsp = false;
    this->running_mv_sum = NULL;
    this->prev_locations = NULL;
    this->curr_locations = NULL;
    this->gop_idx = -1;
    this->gop_pos = 0;
    this->frame_type = 'A';
    this->mv_res_reduction = 8;
    this->iframe_width = -1;
    this->iframe_height = -1;
    this->gop_size = -1;

    memset(&(this->packet), 0, sizeof(this->packet));
    av_init_packet(&(this->packet));
}


void VideoCap::release(void) {
    if (this->img_convert_ctx != NULL) {
        sws_freeContext(this->img_convert_ctx);
        this->img_convert_ctx = NULL;
    }

    if (this->frame != NULL) {
        av_frame_free(&(this->frame));
    }

    if (this->out_frames != NULL) {
        for (int i = 0; i < this->gop_size; i++) {
            if (this->out_frames[i] != NULL) {
                av_frame_free(&(this->out_frames[i]));
            }
        }
        this->out_frames = NULL;
    }
    
    if (this->video_dec_ctx != NULL) {
        avcodec_free_context(&(this->video_dec_ctx));
        this->video_dec_ctx = NULL;
    }

    if (this->fmt_ctx != NULL) {
        avformat_close_input(&(this->fmt_ctx));
        this->fmt_ctx = NULL;
    }

    if (this->opts != NULL) {
        av_dict_free(&(this->opts));
        this->opts = NULL;
    }

    if (this->packet.data) {
        av_packet_unref(&(this->packet));
        this->packet.data = NULL;
    }
    memset(&packet, 0, sizeof(packet));
    av_init_packet(&packet);

    this->codec = NULL;
    this->video_stream = NULL;
    this->video_stream_idx = -1;
    this->frame_number = 0;
    this->is_rtsp = false;

    Py_CLEAR(this->running_mv_sum);
    if (this->prev_locations != NULL){
        free(this->prev_locations);
        this->prev_locations = NULL;
    }
    if (this->curr_locations != NULL){
        free(this->curr_locations);
        this->curr_locations = NULL;
    }
    this->gop_idx = -1;
    this->gop_pos = 0;
    this->frame_type = 'A';
    this->mv_res_reduction = 8;
    this->iframe_width = -1;
    this->iframe_height = -1;
    this->gop_size = -1;
}


bool VideoCap::open(const char *url, char frame_type, int iframe_width, int iframe_height, int mv_res_reduction, int gop_size) {

    bool valid = false;
    AVStream *st = NULL;
    int enc_width, enc_height, idx;

    this->release();

    // if another file is already opened
    if (this->fmt_ctx != NULL)
        goto error;

    this->url = url;

    // open RTSP stream with TCP
    av_dict_set(&(this->opts), "rtsp_transport", "tcp", 0);
    av_dict_set(&(this->opts), "stimeout", "5000000", 0); // set timeout to 5 seconds
    if (avformat_open_input(&(this->fmt_ctx), url, NULL, &(this->opts)) < 0)
        goto error;

    // determine if opened stream is RTSP or not (e.g. a video file)
    // this->is_rtsp = check_format_rtsp(this->fmt_ctx->iformat->name);

    // read packets of a media file to get stream information.
    if (avformat_find_stream_info(this->fmt_ctx, NULL) < 0)
        goto error;

    // find the most suitable stream of given type (e.g. video) and set the codec accordingly
    idx = av_find_best_stream(this->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &(this->codec), 0);
    if (idx < 0)
        goto error;

    // set stream in format context
    this->video_stream_idx = idx;
    st = this->fmt_ctx->streams[this->video_stream_idx];

    // allocate an AVCodecContext and set its fields to default values
    this->video_dec_ctx = avcodec_alloc_context3(this->codec);
    if (!this->video_dec_ctx)
        goto error;

    // fill the codec context based on the values from the supplied codec parameters
    if (avcodec_parameters_to_context(this->video_dec_ctx, st->codecpar) < 0)
        goto error;

    this->video_dec_ctx->thread_count = std::thread::hardware_concurrency();
#ifdef DEBUG
    std::cout << "Using parallel processing with " << this->video_dec_ctx->thread_count << " threads" << std::endl;
#endif

    // backup encoder's width/height
    enc_width = this->video_dec_ctx->width;
    enc_height = this->video_dec_ctx->height;

    // Init the video decoder with the codec and set additional option to extract motion vectors
    av_dict_set(&(this->opts), "flags2", "+export_mvs", 0);
    if (avcodec_open2(this->video_dec_ctx, this->codec, &(this->opts)) < 0)
        goto error;

    this->video_stream = this->fmt_ctx->streams[this->video_stream_idx];

    // checking width/height (since decoder can sometimes alter it, eg. vp6f)
    if (enc_width && (this->video_dec_ctx->width != enc_width))
        this->video_dec_ctx->width = enc_width;
    if (enc_height && (this->video_dec_ctx->height != enc_height))
        this->video_dec_ctx->height = enc_height;
    
    this->frame_type = frame_type;
    if (frame_type == 'I'){
        this->video_dec_ctx->skip_frame = AVDISCARD_NONKEY;
    }

    this->iframe_width = iframe_width;
    this->iframe_height = iframe_height;
    this->mv_res_reduction = mv_res_reduction;
    this->gop_size = std::max(this->gop_size, 1);
    
    // print info (duration, bitrate, streams, container, programs, metadata, side data, codec, time base)
#ifdef DEBUG
    av_dump_format(this->fmt_ctx, 0, url, 0);
#endif

    this->frame = av_frame_alloc();
    if (!this->frame)
        goto error;

    this->out_frames = (AVFrame**) malloc(this->gop_size * sizeof(AVFrame*));
    if (!this->out_frames)
        goto error;
    for (int i = 0; i < this->gop_size; i++) {
        this->out_frames[i] = av_frame_alloc();
        if (!this->out_frames[i])
            goto error;
    }

    if (this->video_stream_idx >= 0)
        valid = true;

error:

    if (!valid)
        this->release();

    return valid;
}


bool VideoCap::grab(void) {

    bool valid = false;
    int got_frame;

    int count_errs = 0;
    const int max_number_of_attempts = 512;

    // make sure file is opened
    if (!this->fmt_ctx || !this->video_stream)
        return false;

    // check if there is a frame left in the stream
    if (this->fmt_ctx->streams[this->video_stream_idx]->nb_frames > 0 &&
        this->frame_number > this->fmt_ctx->streams[this->video_stream_idx]->nb_frames)
        return false;

    // loop over different streams (video, audio) in the file
    while(!valid) {
        av_packet_unref(&(this->packet));

        // read next packet from the stream
        int ret = av_read_frame(this->fmt_ctx, &(this->packet));

        if (ret == AVERROR(EAGAIN))
            continue;

        // if the packet is not from the video stream don't do anything and get next packet
        if (this->packet.stream_index != this->video_stream_idx) {
            av_packet_unref(&(this->packet));
            count_errs++;
            if (count_errs > max_number_of_attempts)
                break;
            continue;
        }

        // decode the video frame
        avcodec_decode_video2(this->video_dec_ctx, this->frame, &got_frame, &(this->packet));

        if(got_frame) {
            this->frame_number++;
            valid = true;
        }
        else {
            count_errs++;
            if (count_errs > max_number_of_attempts)
                break;
        }

    }

    return valid;
}

bool VideoCap::retrieve(AVFrame *out_frame, PyArrayObject **frame, int *step, int *width, int *height, int *cn, char *frame_type, int *gop_idx, int *gop_pos) {
    if (!this->video_stream || !(this->frame->data[0]))
        return false;

    // get frame type (I, P, B, etc.) and create a null terminated c-string
    frame_type[0] = av_get_picture_type_char(this->frame->pict_type);
    frame_type[1] = '\0';

    if (frame_type[0] == 'I'){
        this->gop_idx += 1;
        this->gop_pos = 0;
    } else {
        this->gop_pos += 1;
    }

    if (this->frame_type == 'P' && frame_type[0] != 'P'){
        return this->read(frame, step, width, height, cn, frame_type, gop_idx, gop_pos);
    }

    int new_width = (this->iframe_width > 0) ? this->iframe_width : this->video_dec_ctx->width;
    int new_height = (this->iframe_height > 0) ? this->iframe_height : this->video_dec_ctx->height;

    if (this->img_convert_ctx == NULL ||
        out_frame->width != new_width ||
        out_frame->height != new_height ||
        out_frame->data == NULL) {

        this->img_convert_ctx = sws_getCachedContext(
                this->img_convert_ctx,
                this->video_dec_ctx->width, this->video_dec_ctx->height,
                this->video_dec_ctx->pix_fmt,
                new_width, new_height,
                AV_PIX_FMT_BGR24,
                SWS_BICUBIC,
                NULL, NULL, NULL
                );

        if (this->img_convert_ctx == NULL)
            return false;

        av_frame_unref(out_frame);
        out_frame->format = AV_PIX_FMT_BGR24;
        out_frame->width = new_width;
        out_frame->height = new_height;
        if (0 != av_frame_get_buffer(out_frame, 0))
            return false;
    }

    // change color space of frame
    sws_scale(
        this->img_convert_ctx,
        this->frame->data,
        this->frame->linesize,
        0, this->video_dec_ctx->height,
        out_frame->data,
        out_frame->linesize
        );

    *width = out_frame->width;
    *height = out_frame->height;
    *step = out_frame->linesize[0];
    *cn = 3;

    npy_intp dims[3] = {*height, *width, *cn};
    *frame = (PyArrayObject *)PyArray_SimpleNewFromData(3, dims, NPY_UINT8, out_frame->data[0]);

    *gop_idx = this->gop_idx;
    *gop_pos = this->gop_pos;

    // Copy motion vectors into out_frame
    AVFrameSideData *sd = av_frame_get_side_data(this->frame, AV_FRAME_DATA_MOTION_VECTORS);
    if (sd) {
        if (av_frame_get_side_data(out_frame, AV_FRAME_DATA_MOTION_VECTORS)) {
            av_frame_remove_side_data(out_frame, AV_FRAME_DATA_MOTION_VECTORS);
        }
            
        // AVFrameSideData *out_sd = av_frame_new_side_data(out_frame, AV_FRAME_DATA_MOTION_VECTORS, sd->size);
        // memcpy(out_sd, sd, sd->size);

        if (!av_frame_new_side_data_from_buf(out_frame, AV_FRAME_DATA_MOTION_VECTORS, av_buffer_ref(sd->buf)))
            return false;
    }

    return true;
}

bool VideoCap::read(PyArrayObject **frame, int *step, int *width, int *height, int *cn, char *frame_type, int *gop_idx, int *gop_pos) {
    bool ret = this->grab();
    if (ret)
        ret = this->retrieve(this->out_frames[0], frame, step, width, height, cn, frame_type, gop_idx, gop_pos);
    return ret;
}

bool VideoCap::accumulate(uint8_t **frame, int *step, int *width, int *height, int *cn, char *frame_type, PyArrayObject **accumulated_mv, MVS_DTYPE *num_mvs, int *gop_idx, int *gop_pos) {
    if (!this->video_stream || !(this->frame->data[0]))
        return false;

    // get frame type (I, P, B, etc.) and create a null terminated c-string
    // frame_type[0] = av_get_picture_type_char(this->frame->pict_type);
    // frame_type[1] = '\0';

    // if (frame_type[0] == 'I' || this->prev_locations == NULL || this->curr_locations == NULL || this->running_mv_sum == NULL) {
    //     this->reset_accumulate(&(this->prev_locations), &(this->curr_locations), &(this->running_mv_sum), this->video_dec_ctx->width, this->video_dec_ctx->height);
    // }

    // if (frame_type[0] == 'I'){
    //     this->gop_idx += 1;
    //     this->gop_pos = 0;
    // } else {
    //     this->gop_pos += 1;
    // }

    // if (this->frame_type == 'P' && frame_type[0] != 'P'){
    //     return this->read_accumulate(frame, step, width, height, cn, frame_type, accumulated_mv, num_mvs, gop_idx, gop_pos);
    // }

    // int new_width = (this->iframe_width > 0) ? this->iframe_width : this->video_dec_ctx->width;
    // int new_height = (this->iframe_height > 0) ? this->iframe_height : this->video_dec_ctx->height;

    // if (this->img_convert_ctx == NULL ||
    //     out_frame->width != new_width ||
    //     out_frame->height != new_height ||
    //     out_frame->data == NULL) {

    //     this->img_convert_ctx = sws_getCachedContext(
    //             this->img_convert_ctx,
    //             this->video_dec_ctx->width, this->video_dec_ctx->height,
    //             this->video_dec_ctx->pix_fmt,
    //             new_width, new_height,
    //             AV_PIX_FMT_BGR24,
    //             SWS_BICUBIC,
    //             NULL, NULL, NULL
    //             );

    //     if (this->img_convert_ctx == NULL)
    //         return false;

    //     av_frame_unref(&(this->rgb_frame));
    //     this->rgb_frame.format = AV_PIX_FMT_BGR24;
    //     this->rgb_frame.width = new_width;
    //     this->rgb_frame.height = new_height;
    //     if (0 != av_frame_get_buffer(&(this->rgb_frame), 0))
    //         return false;
    // }

    // // change color space of frame
    // sws_scale(
    //     this->img_convert_ctx,
    //     this->frame->data,
    //     this->frame->linesize,
    //     0, this->video_dec_ctx->height,
    //     this->rgb_frame.data,
    //     this->rgb_frame.linesize
    //     );

    // *frame = out_frame->data[0];
    // *width = out_frame->width;
    // *height = out_frame->height;
    // *step = out_frame->linesize[0];
    // *cn = 3;

    // // get motion vectors
    // AVFrameSideData *sd = av_frame_get_side_data(this->frame, AV_FRAME_DATA_MOTION_VECTORS);
    // if (sd) {
    //     AVMotionVector *mvs = (AVMotionVector *)sd->data;

    //     *num_mvs = sd->size / sizeof(*mvs);

    //     if (*num_mvs > 0) {
    //         int p_dst_x, p_dst_y, p_src_x, p_src_y;
    //         int val_x, val_y;
    //         int original_x, original_y;
    //         const AVMotionVector *mvs = (const AVMotionVector *)sd->data;

    //         int mv_width = this->video_dec_ctx->width / this->mv_res_reduction;
    //         int mv_height = this->video_dec_ctx->height / this->mv_res_reduction;

    //         // #pragma omp parallel for num_threads(std::thread::hardware_concurrency() / 4) \
    //         // private(p_dst_x, p_dst_y, p_src_x, p_src_y, val_x, val_y, original_x, original_y) 
    //         for (int i = 0; i < sd->size / sizeof(*mvs); i++) {
    //             const AVMotionVector *mv = &mvs[i];
    //             val_x = mv->dst_x - mv->src_x;
    //             val_y = mv->dst_y - mv->src_y;
    //             // assert(mv->source == -1);

    //             if (val_x != 0 || val_y != 0) {
    //                 for (int x_start = 0; x_start < mv->w / this->mv_res_reduction; ++x_start) {
    //                     for (int y_start = 0; y_start < mv->h / this->mv_res_reduction; ++y_start) {
    //                         p_dst_x = mv->dst_x / this->mv_res_reduction + x_start - mv->w / (2 * this->mv_res_reduction);
    //                         p_dst_y = mv->dst_y / this->mv_res_reduction + y_start - mv->h / (2 * this->mv_res_reduction);

    //                         p_src_x = mv->src_x / this->mv_res_reduction + x_start - mv->w / (2 * this->mv_res_reduction);
    //                         p_src_y = mv->src_y / this->mv_res_reduction + y_start - mv->h / (2 * this->mv_res_reduction);

    //                         if (p_dst_y >= 0 && p_dst_y < mv_height && 
    //                             p_dst_x >= 0 && p_dst_x < mv_width &&
    //                             p_src_y >= 0 && p_src_y < mv_height && 
    //                             p_src_x >= 0 && p_src_x < mv_width) {
                                
    //                             // Shift macroblock in curr_locations
    //                             original_x = this->prev_locations[p_src_x * mv_height * 2 + p_src_y * 2];
    //                             this->curr_locations[p_dst_x * mv_height * 2 + p_dst_y * 2] = original_x;
                                
    //                             original_y = this->prev_locations[p_src_x * mv_height * 2 + p_src_y * 2 + 1];
    //                             this->curr_locations[p_dst_x * mv_height * 2 + p_dst_y * 2 + 1] = original_y;
                                
    //                             // Accumulate into running_mv_sum the motion vector for the pixels in this macroblock
    //                             // #pragma omp atomic update
    //                             *((npy_int16*)PyArray_GETPTR3(this->running_mv_sum, original_y, original_x, 0)) += val_x;
    //                             // #pragma omp atomic update
    //                             *((npy_int16*)PyArray_GETPTR3(this->running_mv_sum, original_y, original_x, 1)) += val_y;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         memcpy(this->prev_locations, this->curr_locations, mv_width * mv_height * 2 * sizeof(int));
    //     }
    // }
    
    // // Set return value to running_mv_sum
    // *accumulated_mv = this->running_mv_sum;

    // *gop_idx = this->gop_idx;
    // *gop_pos = this->gop_pos;

    return true;
}

bool VideoCap::read_accumulate(uint8_t **frame, int *step, int *width, int *height, int *cn, char *frame_type, PyArrayObject **accumulated_mv, MVS_DTYPE *num_mvs, int *gop_idx, int *gop_pos) {
    bool ret = this->grab();
    // if (ret)
    //     ret = this->retrieve(this->out_frames[0], frame, step, width, height, cn, frame_type, gop_idx, gop_pos);
    if (ret)
        ret = this->accumulate(frame, step, width, height, cn, frame_type, accumulated_mv, num_mvs, gop_idx, gop_pos);
    return ret;
}

// Returns true if the comma-separated list of format names contains "rtsp"
bool VideoCap::check_format_rtsp(const char *format_names) {

    char str[strlen(format_names) + 1];
    strcpy(str, format_names);

    char *format_name;
    char *buffer = str;

    while ((format_name = strtok_r(buffer, ",", &buffer))) {

        if (strcmp(format_name, "rtsp") == 0)
            return true;
    }

    return false;
}

/**
 * Resets prev_locations and curr_locations to coordinate arrays.
 * Resets running_mv_sum to 0.
 * Allocates arrays if they do not yet exist.
 */
void VideoCap::reset_accumulate(int **prev_locations, int **curr_locations, PyArrayObject **running_mv_sum, int width, int height) {
    int h = height / this->mv_res_reduction;
    int w = width / this->mv_res_reduction;

    if (*prev_locations == NULL){
        *prev_locations = (int*) malloc(w * h * 2 * sizeof(int));
    }

    if (*curr_locations == NULL){
        *curr_locations = (int*) malloc(w * h * 2 * sizeof(int));
    }

    // #pragma omp parallel for num_threads(std::thread::hardware_concurrency() / 4)
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            (*prev_locations)[x * h * 2 + y * 2    ]  = x;
            (*prev_locations)[x * h * 2 + y * 2 + 1]  = y;
        }
    }
    memcpy(*curr_locations, *prev_locations, h * w * 2 * sizeof(int));

    if (*running_mv_sum == NULL){
        npy_intp dims[3] = {h, w, 2};
        *running_mv_sum = (PyArrayObject *)PyArray_ZEROS(3, dims, NPY_INT16, 0);
    } else{
        // Set running_mv_sum to 0
        PyArray_FILLWBYTE(*running_mv_sum, 0);
    }
}