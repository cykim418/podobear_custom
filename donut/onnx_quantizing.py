import os
import argparse

from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize(models_path):
    if not os.path.exists(os.path.join(models_path, 'onnx_quantized')):
        os.mkdir(os.path.join(models_path, 'onnx_quantized'))

    onnx_file_list = os.listdir(models_path)

    for model_name in onnx_file_list:
        if '.onnx' in model_name:
            if 'encoder' in model_name:
                model_input = os.path.join(models_path, model_name)
                model_output = os.path.join(models_path, 'onnx_quantized', f'quantized_{model_name}')
                quantize_dynamic(
                    model_input=model_input,
                    model_output=model_output,
                    weight_type=QuantType.QUInt8,
                )
            else:
                model_input = os.path.join(models_path, model_name)
                model_output = os.path.join(models_path, 'onnx_quantized', f'quantized_{model_name}')
                quantize_dynamic(
                    model_input=model_input,
                    model_output=model_output,
                )
    
    if len(os.listdir(os.path.join(models_path, 'onnx_quantized'))) == 3:
        print('Finished Quantizing models')
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_models_path', type=str, required=True)
    args, left_argv = parser.parse_known_args()

    quantize(args.onnx_models_path)


    '''
    후 작업 필요 *** onnx -> ort
    python -m onnxruntime.tools.convert_onnx_models_to_ort <onnx model file or dir>
    '''