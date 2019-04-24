import calibrator
import tensorrt as trt

# Use TensorRT ONNX parser to parse model file, and enable INT8 calibration during engine construction
def build_int8_engine_onnx(model_file, image_dir, batch_size, calibration_batches, engine_file, cache_file='INT8CalibrationTable'):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            parser.parse(model.read())

        # Allow builder to use INT8 or FP16 kernels when building engine
        builder.int8_mode = True
        builder.fp16_mode = True
        calib = calibrator.ONNXEntropyCalibrator(image_dir, batch_size, calibration_batches, cache_file)
        builder.int8_calibrator = calib
        builder.max_batch_size = batch_size

        engine = builder.build_cuda_engine(network)

        with open(engine_file, 'wb') as f:
            f.write(engine.serialize())

