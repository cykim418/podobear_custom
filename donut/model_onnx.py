import torch
import numpy as np
from donut import DonutModel


model = DonutModel.from_pretrained('result/bscard3')

donut_encoder = model.encoder
donut_decoder = model.decoder



'''
Swin Transformer (Donut Encoder) convert to onnx
'''
dummy_donutencoder_input = torch.randn(1, 3, 960, 640)
input_names = ['input_img']
output_names = ['donutencoder_output']

# torch.onnx.export(
#     donut_encoder,
#     dummy_donutencoder_input,
#     "result/donut_encoder.onnx",
#     input_names=input_names,
#     output_names=output_names
# )


import onnxruntime as ort

donut_encoder_ort = ort.InferenceSession('result/donut_encoder.onnx')
donut_encoder_onnx_output = donut_encoder_ort.run(
    None,
    {'input_img':np.random.randn(1,3,960,640).astype(np.float32)}
)

print(donut_encoder_onnx_output[0].shape)
print()



'''
Multilingual BART (Donut Decoder) convert to onnx
'''
# tokenizer = donut_decoder.tokenizer

# dummy_donutdecoder_input_ids = tokenizer('2023_07_21', add_special_tokens=False, return_tensors='pt')['input_ids']
# print(dummy_donutdecoder_input_ids)

dummy_donutdecoder_input_ids = torch.randint(100,(donut_encoder_onnx_output[0].shape[1], 1024), dtype=torch.int32)
dummy_donutdecoder_encoder_hidden_states = torch.randn(donut_encoder_onnx_output[0].shape[1], donut_encoder_onnx_output[0].shape[2])
dummy_donutdecoder_labels = torch.randint(100,(1,donut_encoder_onnx_output[0].shape[1]), dtype=torch.int32)

print('-----------------')
print(dummy_donutdecoder_input_ids.shape)
print(dummy_donutdecoder_encoder_hidden_states.shape)
print(dummy_donutdecoder_labels.shape)


dummy_inputs = (
    {
        'input_ids' : dummy_donutdecoder_input_ids,
        'encoder_hidden_states' : dummy_donutdecoder_encoder_hidden_states,
        'labels' : dummy_donutdecoder_labels
    }
)

torch.onnx.export(
    donut_decoder,
    (dummy_donutdecoder_input_ids, dummy_donutdecoder_encoder_hidden_states, dummy_donutdecoder_labels),
    "result/donut_decoder.onnx",
    input_names=['input_ids', 'encoder_hidden_states', 'labels'],
    output_names=['output1'],
    opset_version=14
)

# import onnxruntime as ort

# donut_decoder_ort = ort.InferenceSession('result/donut_decoder.onnx')
# donut_decoder_onnx_output = donut_decoder_ort.run(
#     None,
#     {'input_ids':np.random.randn(1,600).astype(np.int64),
#      'encoder_hidden_states':np.random.randn(1,600,1024).astype(np.float32),
#      'labels':np.random.randn(1,600).astype(np.int64)}
# )

# print(donut_decoder_onnx_output)


