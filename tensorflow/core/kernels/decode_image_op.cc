/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/image_ops.cc

#include <memory>
#include <limits>
#include <algorithm>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gif/gif_io.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

enum FileFormat {
  kUnknownFormat = 0,
  kPngFormat = 1,
  kJpgFormat = 2,
  kGifFormat = 3,
  kPPMFormat = 4,
  kPGMFormat = 5,
};

// Classify the contents of a file based on starting bytes (the magic number).
FileFormat ClassifyFileFormat(StringPiece data) {
  // The 4th byte of JPEG is '\xe0' or '\xe1', so check just the first three
  if (data.starts_with("\xff\xd8\xff")) return kJpgFormat;
  if (data.starts_with("\x89PNG\r\n\x1a\n")) return kPngFormat;
  if (data.starts_with("\x47\x49\x46\x38")) return kGifFormat;
  if (data.starts_with("P6")) return kPPMFormat;
  if (data.starts_with("P5")) return kPGMFormat;
  return kUnknownFormat;
}

string FileFormatString(FileFormat magic, StringPiece data) {
  switch (magic) {
    case kPngFormat:
      return "PNG";
    case kJpgFormat:
      return "JPEG";
    case kGifFormat:
      return "GIF";
    case kPPMFormat:
      return "PPM";
    case kPGMFormat:
      return "PGM";
    default: {
      if (data.empty()) return "empty file";
      return strings::StrCat("unknown format starting with '",
                             str_util::CEscape(data.substr(0, 16)), "'");
    }
  }
}

void skipComments(StringPiece& input) {
  StringPiece endLine("\n");
  while (input[0] == '#') { // skip comments
    auto endLPos = std::search(input.begin(), input.end(), endLine.begin(), endLine.end());
    input.remove_prefix(endLPos + endLine.size() - input.begin());
  }  
}

// Decode an image (either jpeg, png, or gif).  We use a single op so that
// users don't have to care about which format they have.
class DecodeImageOp : public OpKernel {
 public:
  explicit DecodeImageOp(OpKernelConstruction* context) : OpKernel(context) {
    // Determine which op we are: jpeg, png, gif, or any
    if (type_string() == "DecodeJpeg") {
      format_ = kJpgFormat;
    } else if (type_string() == "DecodeAndCropJpeg") {
      format_ = kJpgFormat;
      flags_.crop = true;
    } else if (type_string() == "DecodePng") {
      format_ = kPngFormat;
    } else if (type_string() == "DecodeGif") {
      format_ = kGifFormat;
    } else if (type_string() == "DecodePPM") {
      format_ = kPPMFormat;
    } else if (type_string() == "DecodePGM") {
      format_ = kPGMFormat;
    } else {
      OP_REQUIRES_OK(context,
                     errors::InvalidArgument("Bad op type ", type_string()));
    }

    if (format_ == kGifFormat || format_ == kPPMFormat) {
      channels_ = 3;
    }else if (format_ == kPGMFormat) {
      channels_ = 1;
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));
      OP_REQUIRES(
          context,
          channels_ == 0 || channels_ == 1 || channels_ == 3 || channels_ == 4,
          errors::InvalidArgument("channels must be 0, 1, 3, or 4, got ",
                                  channels_));
    }
    flags_.components = channels_;

    // In the case of png, we support uint16 output
    if (format_ == kPngFormat || format_ == kPGMFormat || format_ == kPPMFormat) {
      DataType dt;
      OP_REQUIRES_OK(context, context->GetAttr("dtype", &dt));
      OP_REQUIRES(
          context, dt == DataType::DT_UINT8 || dt == DataType::DT_UINT16,
          errors::InvalidArgument("Type must be uint8 or uint16, got ", dt));
      if (dt == DataType::DT_UINT8) {
        channel_bits_ = 8;
      } else {
        channel_bits_ = 16;
      }
    }

    // The TensorFlow-chosen default for jpeg decoding is IFAST, sacrificing
    // image quality for speed.
    flags_.dct_method = JDCT_IFAST;

    if (format_ == kJpgFormat) {
      OP_REQUIRES_OK(context, context->GetAttr("ratio", &flags_.ratio));
      OP_REQUIRES(context,
                  flags_.ratio == 1 || flags_.ratio == 2 || flags_.ratio == 4 ||
                      flags_.ratio == 8,
                  errors::InvalidArgument("ratio must be 1, 2, 4, or 8, got ",
                                          flags_.ratio));
      OP_REQUIRES_OK(context, context->GetAttr("fancy_upscaling",
                                               &flags_.fancy_upscaling));
      OP_REQUIRES_OK(context,
                     context->GetAttr("try_recover_truncated",
                                      &flags_.try_recover_truncated_jpeg));
      OP_REQUIRES_OK(context,
                     context->GetAttr("acceptable_fraction",
                                      &flags_.min_acceptable_fraction));

      string dct_method;
      OP_REQUIRES_OK(context, context->GetAttr("dct_method", &dct_method));
      OP_REQUIRES(
          context,
          (dct_method.empty() || dct_method == "INTEGER_FAST" ||
           dct_method == "INTEGER_ACCURATE"),
          errors::InvalidArgument("dct_method must be one of "
                                  "{'', 'INTEGER_FAST', 'INTEGER_ACCURATE'}"));
      if (dct_method == "INTEGER_FAST") {
        flags_.dct_method = JDCT_IFAST;
      } else if (dct_method == "INTEGER_ACCURATE") {
        flags_.dct_method = JDCT_ISLOW;
      }
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Determine format
    const StringPiece input = contents.scalar<string>()();
    const auto magic = ClassifyFileFormat(input);
    OP_REQUIRES(
        context,
        magic == kJpgFormat || magic == kPngFormat || magic == kGifFormat || 
        magic == kPPMFormat || magic == kPGMFormat,
        errors::InvalidArgument("Expected image (JPEG, PNG, GIF, PGM or PPM), got ",
                                FileFormatString(magic, input)));
    OP_REQUIRES(context, input.size() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument(
                    FileFormatString(magic, input),
                    " contents are too large for int: ", input.size()));
    OP_REQUIRES(context, magic == kPngFormat || magic == kPPMFormat ||
                magic == kPngFormat || channel_bits_ == 8,
                errors::InvalidArgument(FileFormatString(magic, input),
                                        " does not support uint16 output"));

    switch (magic) {
      case kJpgFormat:
        DecodeJpeg(context, input);
        break;
      case kPngFormat:
        DecodePng(context, input);
        break;
      case kGifFormat:
        DecodeGif(context, input);
        break;
      case kPPMFormat:
        DecodePixelMap(context, input);
        break;
      case kPGMFormat:
        DecodePixelMap(context, input);
        break;
      default:
        LOG(FATAL) << "Should never get here after check above";
        break;
    }
  }

  void DecodeJpeg(OpKernelContext* context, StringPiece input) {
    OP_REQUIRES(context, channels_ == 0 || channels_ == 1 || channels_ == 3,
                errors::InvalidArgument(
                    "channels must be 0, 1, or 3 for JPEG, got ", channels_));

    // Use local copy of flags to avoid race condition as the class member is
    // shared among different invocations.
    jpeg::UncompressFlags flags = flags_;
    if (flags.crop) {
      // Update flags to include crop window.
      const Tensor& crop_window = context->input(1);
      OP_REQUIRES(context, crop_window.dims() == 1,
                  errors::InvalidArgument("crop_window must be 1-D, got shape ",
                                          crop_window.shape().DebugString()));
      OP_REQUIRES(context, crop_window.dim_size(0) == 4,
                  errors::InvalidArgument("crop_size must have four elements ",
                                          crop_window.shape().DebugString()));
      auto crop_window_vec = crop_window.vec<int32>();
      flags.crop_y = crop_window_vec(0);
      flags.crop_x = crop_window_vec(1);
      flags.crop_height = crop_window_vec(2);
      flags.crop_width = crop_window_vec(3);
    }

    // Decode jpeg, allocating tensor once the size is known.
    Tensor* output = nullptr;
    OP_REQUIRES(
        context,
        jpeg::Uncompress(
            input.data(), input.size(), flags, nullptr /* nwarn */,
            [=, &output](int width, int height, int channels) -> uint8* {
              Status status(context->allocate_output(
                  0,
                  format_ == kGifFormat
                      ? TensorShape({1, height, width, channels})
                      : TensorShape({height, width, channels}),
                  &output));
              if (!status.ok()) {
                VLOG(1) << status;
                context->SetStatus(status);
                return nullptr;
              }
              return output->flat<uint8>().data();
            }),
        errors::InvalidArgument("Invalid JPEG data or crop window, data size ",
                                input.size()));
  }

  void DecodePng(OpKernelContext* context, StringPiece input) {
    // Start decoding png to get shape details
    png::DecodeContext decode;
    OP_REQUIRES(context,
                png::CommonInitDecode(input, channels_, channel_bits_, &decode),
                errors::InvalidArgument("Invalid PNG header, data size ",
                                        input.size()));

    // Verify that width and height are not too large:
    // - verify width and height don't overflow int.
    // - width can later be multiplied by channels_ and sizeof(uint16), so
    //   verify single dimension is not too large.
    // - verify when width and height are multiplied together, there are a few
    //   bits to spare as well.
    const int width = static_cast<int>(decode.width);
    const int height = static_cast<int>(decode.height);
    const int64 total_size =
        static_cast<int64>(width) * static_cast<int64>(height);
    if (width != static_cast<int64>(decode.width) || width <= 0 ||
        width >= (1LL << 27) || height != static_cast<int64>(decode.height) ||
        height <= 0 || height >= (1LL << 27) || total_size >= (1LL << 29)) {
      png::CommonFreeDecode(&decode);
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("PNG size too large for int: ",
                                          decode.width, " by ", decode.height));
    }

    // Allocate tensor
    Tensor* output = nullptr;
    const auto status = context->allocate_output(
        0,
        format_ == kGifFormat ? TensorShape({1, height, width, decode.channels})
                              : TensorShape({height, width, decode.channels}),
        &output);
    if (!status.ok()) png::CommonFreeDecode(&decode);
    OP_REQUIRES_OK(context, status);

    if (channel_bits_ == 8) {
      // Finish decoding png
      OP_REQUIRES(
          context,
          png::CommonFinishDecode(
              reinterpret_cast<png_bytep>(output->flat<uint8>().data()),
              decode.channels * width * sizeof(uint8), &decode),
          errors::InvalidArgument("Invalid PNG data, size ", input.size()));
    } else {
      // Finish decoding png
      OP_REQUIRES(
          context,
          png::CommonFinishDecode(
              reinterpret_cast<png_bytep>(output->flat<uint16>().data()),
              decode.channels * width * sizeof(uint16), &decode),
          errors::InvalidArgument("Invalid PNG data, size ", input.size()));
    }
  }
  
  void DecodeGif(OpKernelContext* context, StringPiece input) {
    OP_REQUIRES(context, channels_ == 0 || channels_ == 3,
                errors::InvalidArgument("channels must be 0 or 3 for GIF, got ",
                                        channels_));

    // Decode GIF, allocating tensor once the size is known.
    Tensor* output = nullptr;
    OP_REQUIRES(
        context,
        gif::Decode(input.data(), input.size(),
                    [=, &output](int num_frames, int width, int height,
                                  int channels) -> uint8* {
                      Status status;
                      if (format_ == kGifFormat) {
                        status = context->allocate_output(
                            0,
                            TensorShape({num_frames, height, width, channels}),
                            &output);
                      } else if (num_frames == 1) {
                        status = context->allocate_output(
                            0, TensorShape({height, width, channels}), &output);
                      } else {
                        status = errors::InvalidArgument(
                            "Got ", num_frames, " frames, but animated gifs ",
                            "can only be decoded by tf.image.decode_gif or ",
                            "tf.image.decode_image");
                      }
                      if (!status.ok()) {
                        VLOG(1) << status;
                        context->SetStatus(status);
                        return nullptr;
                      }
                      return output->flat<uint8>().data();
                    }),
        errors::InvalidArgument("Invalid GIF data, size ", input.size()));
    }

    // Eigen::Matrix used to handle different format of PixelMap 
    typedef Eigen::Matrix<uint8, 1, Eigen::Dynamic> vecUint8;
    typedef Eigen::Matrix<uint16, 1, Eigen::Dynamic> vecUint16;

    void DecodePixelMap(OpKernelContext* context, StringPiece input) {
      OP_REQUIRES(context, channels_ == 1 || channels_ == 3,
                  errors::InvalidArgument("channels must be 1 or 3 for PixelMap, got ",
                                          channels_));
      Status status;

      if (format_ != ClassifyFileFormat(input)) {
        status = errors::InvalidArgument(
          "File format is ", FileFormatString(ClassifyFileFormat(input), input), 
          " you tried to decode using ", FileFormatString(format_, input), " format");
        
        if (!status.ok()) {
          VLOG(1) << status;
          context->SetStatus(status);
          return;
        }
      }
      /*
      //read header of PixelMap data
      StringPiece endLine("\n");
      StringPiece spacePiece(" ");
      auto endLPos = std::search(input.begin(), input.end(), endLine.begin(), endLine.end());
      auto spacePos = std::search(input.begin(), input.end(), spacePiece.begin(), spacePiece.end());
      
      input.remove_prefix(endLPos + endLine.size() - input.begin());
    
      skipComments(input);
      
      // Get width and height
      auto nPos = input.find(' ');
      auto width = stoi(input.substr(0, nPos).ToString());
      input.remove_prefix(nPos + 1);
      endLPos = std::search(input.begin(), input.end(), endLine.begin(), endLine.end());
      auto height = stoi(input.substr(0, std::distance(input.begin(), endLPos)).ToString());
      input.remove_prefix(endLPos + endLine.size() - input.begin());
    
      skipComments(input);
      
      // Get maximum value of pixels in image 
      endLPos = std::search(input.begin(), input.end(), endLine.begin(), endLine.end());
      auto maxVal = stoi(input.substr(0, std::distance(input.begin(), endLPos)).ToString());
      input.remove_prefix(endLPos + endLine.size() - input.begin());
      
      skipComments(input);
      if (maxVal <= 0) {
        status = errors::InvalidArgument(
          "Got 0 as maximum value for PPM/PGM decode ",
          "metadata might be corrupted");
        
        if (!status.ok()) {
          VLOG(1) << status;
          context->SetStatus(status);
          return;
        }
      }
      */
      //read header of PixelMap data
      auto nPos = input.find('\n');
      auto sPos = input.find(' ');
      input.remove_prefix(std::min(nPos, sPos) + 1);
    
      skipComments(input);
      
      // Get width and height
      nPos = input.find('\n');
      sPos = input.find(' ');
      auto width = stoi(input.substr(0, std::min(nPos, sPos)).ToString());
      input.remove_prefix(std::min(nPos, sPos) + 1);
      skipComments(input);

      nPos = input.find('\n');
      sPos = input.find(' ');
      auto height = stoi(input.substr(0, std::min(nPos, sPos)).ToString());
      input.remove_prefix(std::min(nPos, sPos) + 1);
      skipComments(input);
      
      // Get maximum value of pixels in image 
      nPos = input.find('\n');
      sPos = input.find(' ');
      auto maxVal = stoi(input.substr(0, std::min(nPos, sPos)).ToString());
      input.remove_prefix(std::min(nPos, sPos) + 1);
      skipComments(input);

      if (maxVal <= 0) {
        status = errors::InvalidArgument(
          "Got 0 as maximum value for PPM/PGM decode ",
          "metadata might be corrupted");
        
        if (!status.ok()) {
          VLOG(1) << status;
          context->SetStatus(status);
          return;
        }
      }

      Tensor* output = nullptr;
      status = context->allocate_output(
        0,
        TensorShape({height, width, channels_}),
        &output);
      float sizeRatio = (float) input.size() / (float) (height*width*channels_);
      if (sizeRatio == 2) {
        memcpy(output->flat<uint16>().data(), input.data(), height*width*channels_*sizeof(uint16));
        if (maxVal != std::numeric_limits<uint16>::max()) {
          auto eigenOut = Eigen::Map<vecUint16>(output->flat<uint16>().data(), height*width*channels_);
          eigenOut /= maxVal;
          eigenOut *= std::numeric_limits<uint16>::max(); 
        }
      } else if (sizeRatio == 1) {
        memcpy(output->flat<uint8>().data(), input.data(), height*width*channels_*sizeof(uint8));
        
        if (maxVal != std::numeric_limits<uint8>::max()) {
          auto eigenOut = Eigen::Map<vecUint8>(output->flat<uint8>().data(), height*width*channels_);
          eigenOut /= maxVal;
          eigenOut *= std::numeric_limits<uint8>::max(); 
        }
      } else {
        status = errors::InvalidArgument(
          "Input data size mismatched with metadata informations input.size (",
          input.size(), ") != width*height*channels (", width, "*", height, "*",
          channels_, "=", height*width*channels_, ") your file data might be",
          "corrupted");
        if (!status.ok()) {
          VLOG(1) << status;
          context->SetStatus(status);
          return;
        }
      }
    }

 private:
  FileFormat format_;
  int channels_;
  int channel_bits_ = 8;
  jpeg::UncompressFlags flags_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeJpeg").Device(DEVICE_CPU), DecodeImageOp);
REGISTER_KERNEL_BUILDER(Name("DecodePng").Device(DEVICE_CPU), DecodeImageOp);
REGISTER_KERNEL_BUILDER(Name("DecodeGif").Device(DEVICE_CPU), DecodeImageOp);
REGISTER_KERNEL_BUILDER(Name("DecodePPM").Device(DEVICE_CPU), DecodeImageOp);
REGISTER_KERNEL_BUILDER(Name("DecodePGM").Device(DEVICE_CPU), DecodeImageOp);
REGISTER_KERNEL_BUILDER(Name("DecodeAndCropJpeg").Device(DEVICE_CPU),
                        DecodeImageOp);

}  // namespace
}  // namespace tensorflow
