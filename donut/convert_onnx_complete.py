import torch
from donut import DonutModel
import onnxruntime as ort
import numpy as np
import PIL
from PIL import ImageOps
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import List

import functools
import operator
import warnings
warnings.filterwarnings('ignore')




# DonutDecoder Without Past key vlaues
class BARTDecoderWithoutPast(torch.nn.Module):
    def __init__(self,  decoder, tokenizer, config):
        super().__init__()
        self.decoder_model = decoder
        self.tokenizer = tokenizer
        self.config = config

        self.decoder_model.forward = self.forward
        self.decoder_model.config.is_encoder_decoder = True
        self.add_special_tokens(["<sep/>"])
        self.decoder_model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
    

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.decoder_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        outputs = self.decoder_model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )
        logits = self.decoder_model.lm_head(outputs[0])

        return logits, outputs[1]
    

class BARTDecoderWithPast(torch.nn.Module):
    def __init__(self,  decoder, tokenizer, config):
        super().__init__()
        self.decoder_model = decoder
        self.tokenizer = tokenizer
        self.config = config

        self.decoder_model.forward = self.forward
        self.decoder_model.config.is_encoder_decoder = True
        self.add_special_tokens(["<sep/>"])
        self.decoder_model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id


    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))



    def forward(self, *inputs):
        input_ids , attention_mask, encoder_hidden_states = inputs[:3]
        list_pkv = inputs[3:]
        past_key_values = tuple(list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4))

        outputs = self.decoder_model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values
        )
        logits = self.decoder_model.lm_head(outputs[0])

        return logits, outputs[1]

    


def generate_onnx(model):
    # DonutModel의 encoder와 decoder를 추출
    Donut_encoder = model.encoder
    Donut_decoder = model.decoder

    decoder = BARTDecoderWithoutPast(Donut_decoder.model, Donut_decoder.tokenizer, Donut_decoder.model.config)
    decoder_with_past = BARTDecoderWithPast(Donut_decoder.model, Donut_decoder.tokenizer, Donut_decoder.model.config)

    decoder_model_config = Donut_decoder.model.config
    
    # create dummy inputs
    tokenizer = Donut_decoder.tokenizer
    sample_input = "<s><s_2023_07_21>"
    model_inputs = tokenizer(sample_input, add_special_tokens=False, return_tensors='pt')
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']

    n_heads = decoder_model_config.decoder_attention_heads
    seqence_length_a = input_ids.shape[0]
    seqence_length_b = 600 
    d_kv = decoder_model_config.d_model // n_heads

    input_ids_dec = torch.ones((1,1), dtype=torch.int64)
    attention_mask_dec = torch.ones((1,seqence_length_a), dtype=torch.int64)
    enc_out = torch.ones(
        (1, 600, decoder_model_config.d_model), dtype=torch.float32)
    sa = torch.ones(
        (1, n_heads, seqence_length_a, d_kv), dtype=torch.float32)
    ca = torch.ones(
        (1, n_heads, 600, d_kv), dtype=torch.float32)
    
    attention_block = (sa, sa, ca, ca)
    past_key_values = (attention_block,) * model.config.decoder_layer
    flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

    decoder_all_inputs = tuple(
        [input_ids_dec, attention_mask, enc_out] + flat_past_key_values)
    num_of_inputs = 4* model.config.decoder_layer


    # Exports to ONNX
    with torch.no_grad():
        
        # export decoder
        decoder_inputs = ["input_ids", "attention_mask", "encoder_hidden_states"]
        pkv_input_names = [
            f"l{i//4}_{'self' if i % 4 < 2 else 'cross'}_{'keys' if not i % 2 else 'vals'}"
            for i in range(num_of_inputs)
        ]
        decoder_input_names = decoder_inputs + pkv_input_names

        decoder_outputs = ['logits']
        pkv_output_names = [
            f"l{i//4}_{'self_out' if i % 4 < 2 else 'cross'}_{'keys' if not i % 2 else 'vals'}"
            for i in range(num_of_inputs)
        ]
        decoder_output_names = decoder_outputs + pkv_output_names

        dyn_axis = {
            "input_ids": {0: "batch", 1: "seq_length"},
            "attention_mask": {0: "batch", 1: "seq_length"},
            "encoder_hidden_states": {0: "batch"},
            "logits": {0: "batch", 1: "seq_length"},
        }
        dyn_pkv = {
            name: {0: "batch", 2: "seq_length"} for name in pkv_input_names + pkv_output_names
        }
        dyn_axis_params = {**dyn_axis, **dyn_pkv}

        torch.onnx.export(
            decoder_with_past,
            decoder_all_inputs,
            'result/decoder_with_past.onnx',
            export_params=True,
            do_constant_folding=False,
            opset_version=12,
            input_names=decoder_input_names,
            output_names=decoder_output_names,
            dynamic_axes=dyn_axis_params
        )


        # initial decoder to produce past key values
        torch.onnx.export(
            decoder,
            (input_ids_dec, attention_mask_dec, enc_out),
            'result/decoder.onnx',
            export_params=True,
            do_constant_folding=False,
            opset_version=12,
            input_names=[
                "input_ids",
                "attention_mask",
                "encoder_hidden_states",
            ],
            output_names=decoder_output_names,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_length"},
                "attention_mask": {0: "batch", 1: "seq_length"},
                "encoder_hidden_states": {0: "batch"},
                "logits": {0: "batch", 1: "seq_length"},
                **{
                    name: {0: "batch", 2: "seq_length"} for name in pkv_output_names
                }
            },
        )


        # encoder
        torch.onnx.export(
            Donut_encoder,
            torch.ones((1,3,960,640)),
            'result/encoder.onnx',
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["hidden_states"],
            dynamic_axes={
                "image": {0: "batch", 2: "h", 3: "w"},
                "hidden_states": {0: "batch"},
            },
        )



donut_model = DonutModel.from_pretrained('result/bscard3')
# generate_onnx(donut_model)


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

#
# encoder_sess = ort.InferenceSession('result/encoder.onnx')

# decoder_sess1 = ort.InferenceSession('result/decoder.onnx')
# decoder_sess2 = ort.InferenceSession('result/decoder_with_past.onnx')


# 경량화 모델
encoder_sess = ort.InferenceSession('result/result_model0.onnx')
decoder_sess1 = ort.InferenceSession('result/result_model1.onnx')
decoder_sess2 = ort.InferenceSession('result/result_model2.onnx')

tokenizer = donut_model.decoder.tokenizer
test_image = PIL.Image.open('hhg_enssel.jpg')
encoder_input = donut_model.encoder.prepare_input(img=test_image).unsqueeze(0).numpy()
encoder_output = encoder_sess.run(None, {'image': encoder_input})
print(encoder_output[0].shape)



prompt = '<s_2023_07_21>'
print(f'Start Token: {prompt}')
promt_ids = tokenizer(f'{prompt}', add_special_tokens=False, return_tensors="pt")




res = []
for i in range(120):
    if i == 0:
        res.append(prompt)

        decoder_output_logits1 = decoder_sess1.run(None, 
                                            {'input_ids': promt_ids['input_ids'].numpy(), 
                                            'attention_mask':promt_ids['attention_mask'].numpy(), 
                                            'encoder_hidden_states':encoder_output[0]})


        list_pkv = tuple(torch.from_numpy(x) for x in decoder_output_logits1[1:])
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
        output_probs = softmax(decoder_output_logits1[0][0])

        # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
        top_token = np.argmax(output_probs, axis=-1)

        # 생성된 토큰들을 문자열로 변환합니다.
        generated_text = tokenizer.decode(top_token[0])
        res.append(generated_text)
        # prompt += generated_text
        print(f'Generated Token: {generated_text}')
    
    else:
        # promt_ids = tokenizer(f'{prompt}', add_special_tokens=False, return_tensors="pt")
        attention_mask = [[1 for _ in range(i+1)]]
        print("attention Mask -> ",attention_mask)
        print('input_ids', top_token)

        decoder_inputs = {
            "input_ids": np.array([[top_token[0]]]),
            "attention_mask": np.array(attention_mask),
        }
        
        flat_past_key_values = functools.reduce(operator.iconcat, out_past_key_values, [])
        
        input_names = [x.name for x in decoder_sess2.get_inputs()]
        inputs = [
             np.array([[top_token[0]]]).astype(np.int64),
             np.array(attention_mask).astype(np.int64), 
        ] + [
            tensor.numpy() for tensor in flat_past_key_values
        ]
        decoder_inputs = dict(zip(input_names, inputs))

        decoder_output_logits2 = decoder_sess2.run(None, decoder_inputs)
        

        list_pkv = tuple(torch.from_numpy(x) for x in decoder_output_logits2[1:])
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
        output_probs = softmax(decoder_output_logits2[0][0])

        # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
        top_token = np.argmax(output_probs, axis=-1)

        # 생성된 토큰들을 문자열로 변환합니다.
        generated_text = tokenizer.decode(top_token[0])
        res.append(generated_text)
        prompt += generated_text
        print(f'Generated Token: {generated_text}')
    
print(res)
