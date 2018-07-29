#include "./convolution.h"
#include "../layer/convolution.h"


namespace singa {

ConvHandle::ConvHandle(const Tensor &input,
                       const std::vector<size_t>& kernel_size,
                       const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                       const size_t in_channels, const size_t out_channels,
                       const bool bias) {
  kernel_h = kernel_size[0];
  kernel_w = kernel_size[1];

  pad_h = padding[0];
  pad_w = padding[1];

  stride_h = stride[0];
  stride_w = stride[1];

  channels = in_channels;
  num_filters = out_channels;

  bias_term = bias;

  batchsize = input.shape(0);
  CHECK(input.shape(1) == in_channels) <<
                                       "the number of input channels mismatched.";
  height = input.shape(2);
  width = input.shape(3);

  conv_height = 1;
  if (stride_h > 0)
    conv_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  conv_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  col_height = in_channels * kernel_w * kernel_h;
  col_width = conv_height * conv_width;
  imagesize = input.Size() / batchsize;
}



Tensor CpuConvForward(const Tensor &x, Tensor &W,  Tensor &b,
                      const ConvHandle &ch) {
  CHECK_EQ(x.device()->lang(), kCpp);

  CHECK(x.shape(1) == ch.channels && x.shape(2) == ch.height &&
        x.shape(3) == ch.width) << "input sample shape should not change";

  CHECK(W.shape(0) == ch.num_filters && W.shape(1) == ch.channels &&
        W.shape(2) == ch.kernel_h
        && W.shape(3) == ch.kernel_w) << "weights shape should not change";

  Shape w_shape = W.shape();
  Shape b_shape;
  if (ch.bias_term)
    b_shape = b.shape();

  W.Reshape(Shape{ch.num_filters, ch.col_height});
  if (ch.bias_term)
    b.Reshape(Shape{ch.num_filters});

  DataType dtype = x.data_type();
  auto dev = x.device();
  Shape shape{ch.batchsize, ch.num_filters, ch.conv_height, ch.conv_width};
  Tensor output(shape, dev, dtype);

  Tensor col_data(Shape{ch.col_height, ch.col_width});//broadcasted image

  float *data_col = new float[ch.col_height * ch.col_width];
  auto in_data = x.data<float>();
  for (size_t num = 0; num < ch.batchsize; num++) {
    Im2col(in_data + num * ch.imagesize, ch.channels, ch.height, ch.width,
           ch.kernel_h,
           ch.kernel_w, ch.pad_h, ch.pad_w, ch.stride_h, ch.stride_w, data_col);

    col_data.CopyDataFromHostPtr(data_col, ch.col_height * ch.col_width);
    Tensor each = Mult(W, col_data);
    if (ch.bias_term) {
      AddColumn(b, &each);
    }
    CopyDataToFrom(&output, each, each.Size(), num * each.Size());
  };
  W.Reshape(w_shape);
  if (ch.bias_term)
    b.Reshape(b_shape);
  return output;
}

Tensor CpuConvBackwardx(const Tensor &dy, Tensor &W, const Tensor &x,
                        const ConvHandle &ch) {
  CHECK_EQ(dy.device()->lang(), kCpp);

  CHECK(dy.shape(1) == ch.num_filters && dy.shape(2) == ch.conv_height &&
        dy.shape(3) == ch.conv_width) << "input gradients shape should not change";

  CHECK(W.shape(0) == ch.num_filters && W.shape(1) == ch.channels &&
        W.shape(2) == ch.kernel_h
        && W.shape(3) == ch.kernel_w) << "weights shape should not change";

  Shape w_shape = W.shape();
  W.Reshape(Shape{ch.num_filters, ch.col_height});

  Tensor dx;
  dx.ResetLike(x);

  float *dx_b = new float[ch.imagesize];

  for (size_t num = 0; num < ch.batchsize; num++) {
    Tensor grad_b(Shape{ch.num_filters, ch.conv_height * ch.conv_width});
    CopyDataToFrom(&grad_b, dy, grad_b.Size(), 0, num * grad_b.Size());
    Tensor dcol_b = Mult(Transpose(W), grad_b);
    auto dcol_data = dcol_b.data<float>();
    Col2im(dcol_data, ch.channels, ch.height, ch.width, ch.kernel_h, ch.kernel_w,
           ch.pad_h,
           ch.pad_w, ch.stride_h, ch.stride_w, dx_b);
    dx.CopyDataFromHostPtr(dx_b, ch.imagesize, num * ch.imagesize);
  }
  W.Reshape(w_shape);
  return dx;
}

Tensor CpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W,
                        const ConvHandle &ch) {
  CHECK_EQ(dy.device()->lang(), kCpp);

  CHECK(dy.shape(1) == ch.num_filters && dy.shape(2) == ch.conv_height &&
        dy.shape(3) == ch.conv_width) << "input gradients shape should not change";

  CHECK(x.shape(1) == ch.channels && x.shape(2) == ch.height &&
        x.shape(3) == ch.width) << "input sample shape should not change";

  Tensor dW;
  dW.ResetLike(W);
  dW.SetValue(0.0f);

  Shape w_shape = W.shape();
  dW.Reshape(Shape{ch.num_filters, ch.col_height});

  Tensor col_data(Shape{ch.col_height, ch.col_width});//broadcasted image

  float *data_col = new float[ch.col_height * ch.col_width];
  auto in_data = dy.data<float>();
  for (size_t num = 0; num < ch.batchsize; num++) {
    Im2col(in_data + num * ch.imagesize, ch.channels, ch.height, ch.width,
           ch.kernel_h,
           ch.kernel_w, ch.pad_h, ch.pad_w, ch.stride_h, ch.stride_w, data_col);
    col_data.CopyDataFromHostPtr(data_col, ch.col_height * ch.col_width);
    Tensor grad_b(Shape{ch.num_filters, ch.conv_height * ch.conv_width});
    CopyDataToFrom(&grad_b, dy, grad_b.Size(), 0, num * grad_b.Size());
    dW += Mult(grad_b, Transpose(col_data));
  }
  dW.Reshape(w_shape);
  return dW;
}

Tensor CpuConvBackwardb(const Tensor &dy, const Tensor &b,
                        const ConvHandle &ch) {
  CHECK_EQ(dy.device()->lang(), kCpp);

  CHECK(dy.shape(1) == ch.num_filters && dy.shape(2) == ch.conv_height &&
        dy.shape(3) == ch.conv_width) << "input gradients shape should not change";

  CHECK(b.shape(0) == ch.num_filters) << "bias shape should not change";

  Tensor db;
  db.ResetLike(b);

  auto tmpshp = Shape{ch.batchsize * ch.num_filters, dy.Size() / (ch.batchsize * ch.num_filters)};
  Tensor tmp1 = Reshape(dy, tmpshp);

  Tensor tmp2(Shape{ch.batchsize * ch.num_filters});
  SumColumns(tmp1, &tmp2);
  Tensor tmp3 = Reshape(tmp2, Shape{ch.batchsize, ch.num_filters});

  SumRows(tmp3, &db);

  return db;
};

#ifdef USE_CUDNN
CudnnConvHandle::CudnnConvHandle(const Tensor &input,
                                 const std::vector<size_t>& kernel_size,
                                 const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                                 const size_t in_channels, const size_t out_channels, const bool bias,
                                 const size_t workspace_byte_limit, const std::string& prefer)
  : ConvHandle(input, kernel_size, stride, padding, in_channels, out_channels,
               bias) {

  DataType dtype = input.data_type();
  auto dev = input.device();
  Context *ctx = dev->context(0);

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  if (bias_term)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));


  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels, height, width));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                y_desc, CUDNN_TENSOR_NCHW, GetCudnnDataType(dtype), batchsize,
                num_filters, conv_height, conv_width));
  if (bias_term)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                           GetCudnnDataType(dtype), 1,
                                           num_filters, 1, 1));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w,
              stride_h, stride_w, 1, 1,
              CUDNN_CROSS_CORRELATION
#if CUDNN_MAJOR >= 7
              , GetCudnnDataType(dtype)
#endif
                                             ));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, GetCudnnDataType(dtype),
                                         CUDNN_TENSOR_NCHW, num_filters,
                                         channels, kernel_h, kernel_w));
  if (prefer == "fastest" || prefer == "limited_workspace" ||
      prefer == "no_workspace") {
    cudnnConvolutionFwdPreference_t fwd_pref;
    cudnnConvolutionBwdFilterPreference_t bwd_filt_pref;
    cudnnConvolutionBwdDataPreference_t bwd_data_pref;
    if (prefer == "fastest") {
      fwd_pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
    } else if (prefer == "limited_workspace") {
      fwd_pref = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
    } else {
      fwd_pref = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
      bwd_filt_pref = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
      bwd_data_pref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
    }
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
                  ctx->cudnn_handle, x_desc, filter_desc, conv_desc, y_desc, fwd_pref,
                  workspace_byte_limit, &fp_alg));
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
                  ctx->cudnn_handle, x_desc, y_desc, conv_desc, filter_desc,
                  bwd_filt_pref, workspace_byte_limit, &bp_filter_alg));
    // deprecated in cudnn v7
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
                  ctx->cudnn_handle, filter_desc, y_desc, conv_desc, x_desc,
                  bwd_data_pref, workspace_byte_limit, &bp_data_alg));
  } else if (prefer == "autotune") {
    const int topk = 1;
    int num_fp_alg, num_bp_filt_alg, num_bp_data_alg;
    cudnnConvolutionFwdAlgoPerf_t fp_algperf[topk];
    cudnnConvolutionBwdFilterAlgoPerf_t bp_filt_perf[topk];
    cudnnConvolutionBwdDataAlgoPerf_t bp_data_perf[topk];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
                  ctx->cudnn_handle, x_desc, filter_desc, conv_desc, y_desc, topk,
                  &num_fp_alg, fp_algperf));
    fp_alg = fp_algperf[0].algo;
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
                  ctx->cudnn_handle, x_desc, y_desc, conv_desc, filter_desc, topk,
                  &num_bp_filt_alg, bp_filt_perf));
    bp_filter_alg = bp_filt_perf[0].algo;
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
                  ctx->cudnn_handle, filter_desc, y_desc, conv_desc, x_desc, topk,
                  &num_bp_data_alg, bp_data_perf));
    bp_data_alg = bp_data_perf[0].algo;
  } else {
    LOG(FATAL) << "Preferred algorithm is not available!";
  }

  size_t fp_byte, bp_data_byte, bp_filter_byte;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                ctx->cudnn_handle, x_desc, filter_desc, conv_desc, y_desc, fp_alg,
                &fp_byte));
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                ctx->cudnn_handle, filter_desc, y_desc, conv_desc, x_desc,
                bp_data_alg, &bp_data_byte));
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                ctx->cudnn_handle, x_desc, y_desc, conv_desc, filter_desc,
                bp_filter_alg, &bp_filter_byte));
  workspace_count = std::max(std::max(fp_byte, bp_data_byte), bp_filter_byte) /
                    sizeof(float) +
                    1;
  if (workspace_count * sizeof(float) > workspace_byte_limit)
    LOG(WARNING) << "The required memory for workspace ("
                 << workspace_count * sizeof(float)
                 << ") is larger than the expected Bytes ("
                 << workspace_byte_limit << ")";
  workspace = Tensor(Shape{workspace_count}, dev, dtype);
}

CudnnConvHandle::~CudnnConvHandle() {
  if (bias_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
  if (filter_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
  if (conv_desc != nullptr)
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
  if (x_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  if (y_desc != nullptr) CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
}

Tensor GpuConvForward(const Tensor &x, const Tensor &W, const Tensor &b,
                      const CudnnConvHandle &cch) {
  CHECK_EQ(x.device()->lang(), kCuda);

  DataType dtype = x.data_type();
  auto dev = x.device();

  Shape shape{cch.batchsize, cch.num_filters, cch.conv_height, cch.conv_width};
  Tensor output(shape, dev, dtype);

  output.device()->Exec([&output, &x, &W, &cch](Context * ctx) {
    Block *inblock = x.block(), *outblock = output.block(),
           *wblock = W.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionForward(ctx->cudnn_handle, &alpha, cch.x_desc,
                            inblock->data(), cch.filter_desc, wblock->data(),
                            cch.conv_desc, cch.fp_alg,
                            cch.workspace.block()->mutable_data(),
                            cch.workspace_count * sizeof(float), &beta,
                            cch.y_desc, outblock->mutable_data());
  }, {x.block(), W.block()}, {output.block()}, cch.workspace.block());

  if (cch.bias_term) {
    output.device()->Exec([&output, &b, &cch](Context * ctx) {
      float beta = 1.f, alpha = 1.0f;
      Block *outblock = output.block(), *bblock = b.block();
      cudnnAddTensor(ctx->cudnn_handle, &alpha, cch.bias_desc,
                     bblock->data(), &beta, cch.y_desc,
                     outblock->mutable_data());
    }, {output.block(), b.block()}, {output.block()});
  }

  return output;
}

Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x,
                        const CudnnConvHandle &cch) {
  CHECK_EQ(dy.device()->lang(), kCuda);

  Tensor dx;
  dx.ResetLike(x);

  dy.device()->Exec([&dx, &dy, &W, &cch](Context * ctx) {
    Block *wblock = W.block(), *dyblock = dy.block(),
           *dxblock = dx.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardData(ctx->cudnn_handle, &alpha, cch.filter_desc,
                                 wblock->data(), cch.y_desc, dyblock->data(),
                                 cch.conv_desc, cch.bp_data_alg,
                                 cch.workspace.block()->mutable_data(),
                                 cch.workspace_count * sizeof(float), &beta,
                                 cch.x_desc, dxblock->mutable_data());
  }, {dy.block(), W.block()}, {dx.block(), cch.workspace.block()});

  return dx;
}

Tensor GpuConvBackwardW(const Tensor &dy, const Tensor &x, const Tensor &W,
                        const CudnnConvHandle &cch) {
  CHECK_EQ(dy.device()->lang(), kCuda);

  Tensor dW;
  dW.ResetLike(W);

  dy.device()->Exec([&dW, &dy, &x, &cch](Context * ctx) {
    Block *inblock = x.block(), *dyblock = dy.block(),
           *dwblock = dW.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardFilter(
      ctx->cudnn_handle, &alpha, cch.x_desc, inblock->data(),
      cch.y_desc, dyblock->data(), cch.conv_desc, cch.bp_filter_alg,
      cch.workspace.block()->mutable_data(),
      cch.workspace_count * sizeof(float), &beta, cch.filter_desc,
      dwblock->mutable_data());
  }, {dy.block(), x.block()}, {dW.block(), cch.workspace.block()});

  return dW;
}

// input Tensor b for Reset db purpose, can avoid this later.
Tensor GpuConvBackwardb(const Tensor &dy, const Tensor &b,
                        const CudnnConvHandle &cch) {
  CHECK_EQ(dy.device()->lang(), kCuda);

  Tensor db;
  db.ResetLike(b);

  dy.device()->Exec([&db, &dy, &cch](Context * ctx) {
    Block *dyblock = dy.block(), *dbblock = db.block();
    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardBias(ctx->cudnn_handle, &alpha, cch.y_desc,
                                 dyblock->data(), &beta, cch.bias_desc,
                                 dbblock->mutable_data());
  }, {dy.block()}, {db.block()});

  return db;
}
#endif  // USE_CUDNN

}  // namespace singa
