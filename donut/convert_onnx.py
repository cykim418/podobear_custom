import os
import json
import functools
import operator
import argparse
from typing import List

import PIL
import torch
import numpy as np
import onnxruntime as ort

from donut import DonutModel

"""
This converts Encoder and Decoder of the model to ONNX separately.

Need to install
onnx==1.14.1
onnxruntime==1.15.1
"""


# DonutDecoder Without Past key values
class BARTDecoderWithoutPast(torch.nn.Module):
    def __init__(self, decoder, tokenizer, config):
        super().__init__()
        self.decoder_model = decoder
        self.tokenizer = tokenizer
        self.config = config

        self.decoder_model.forward = self.forward
        self.decoder_model.config.is_encoder_decoder = True
        self.add_special_tokens(["<sep/>"])
        self.decoder_model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id

    def add_special_tokens(self, list_of_tokens: List[str]):
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


# DonutDecoder With Past key values
class BARTDecoderWithPast(torch.nn.Module):
    def __init__(self, decoder, tokenizer, config):
        super().__init__()
        self.decoder_model = decoder
        self.tokenizer = tokenizer
        self.config = config

        self.decoder_model.forward = self.forward
        self.decoder_model.config.is_encoder_decoder = True
        self.add_special_tokens(["<sep/>"])
        self.decoder_model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id

    def add_special_tokens(self, list_of_tokens: List[str]):
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, *inputs):
        input_ids, attention_mask, encoder_hidden_states = inputs[:3]
        list_pkv = inputs[3:]
        past_key_values = tuple(list_pkv[i: i + 4] for i in range(0, len(list_pkv), 4))

        outputs = self.decoder_model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values
        )
        logits = self.decoder_model.lm_head(outputs[0])

        return logits, outputs[1]


# Generta funtion
def generate_onnx(model, save_path, config_path):
    # Load Donut Model config
    with open(os.path.join(config_path, 'config.json'), 'r') as f:
        config = json.load(f)
    json.dumps(config)

    # Encoder convert to ONNX
    encoder = model.encoder
    encoder_dummy_inputs = torch.ones((1, 3, config['input_size'][0], config['input_size'][1]))

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            encoder_dummy_inputs,
            os.path.join(save_path, 'onnx', 'encoder.onnx'),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["hidden_states"],
            dynamic_axes={
                "image": {0: "batch", 2: "h", 3: "w"},
                "hidden_states": {0: "batch"},
            },
        )

    encoder_sess = ort.InferenceSession(os.path.join(save_path, 'onnx', 'encoder.onnx'))
    encoder_output = encoder_sess.run(None, {'image': encoder_dummy_inputs.numpy()})

    # Decoder convert ot ONNX
    Donut_decoder = model.decoder

    decoder = BARTDecoderWithoutPast(Donut_decoder.model, Donut_decoder.tokenizer, Donut_decoder.model.config)
    decoder_with_past = BARTDecoderWithPast(Donut_decoder.model, Donut_decoder.tokenizer, Donut_decoder.model.config)

    decoder_model_config = Donut_decoder.model.config

    # create decoder dummy inputs
    tokenizer = Donut_decoder.tokenizer
    sample_input = "<s><s_2023_07_21>"
    model_inputs = tokenizer(sample_input, add_special_tokens=False, return_tensors='pt')
    input_ids = model_inputs['input_ids'].type(torch.int32)
    attention_mask = model_inputs['attention_mask'].type(torch.int32)

    n_heads = decoder_model_config.decoder_attention_heads
    seqence_length_a = input_ids.shape[0]
    seqence_length_b = encoder_output[0].shape[1]
    d_kv = decoder_model_config.d_model // n_heads

    input_ids_dec = torch.ones((1, 1), dtype=torch.int32)
    attention_mask_dec = torch.ones((1, seqence_length_a), dtype=torch.int32)
    enc_out = torch.ones(
        (1, seqence_length_b, decoder_model_config.d_model), dtype=torch.float32)
    sa = torch.ones(
        (1, n_heads, seqence_length_a, d_kv), dtype=torch.float32)
    ca = torch.ones(
        (1, n_heads, seqence_length_b, d_kv), dtype=torch.float32)

    attention_block = (sa, sa, ca, ca)
    past_key_values = (attention_block,) * model.config.decoder_layer
    flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

    decoder_all_inputs = tuple(
        [input_ids_dec, attention_mask, enc_out] + flat_past_key_values)
    num_of_inputs = 4 * model.config.decoder_layer

    # Decoder convert to ONNX
    with torch.no_grad():
        decoder_inputs = ["input_ids", "attention_mask", "encoder_hidden_states"]
        pkv_input_names = [
            f"l{i // 4}_{'self' if i % 4 < 2 else 'cross'}_{'keys' if not i % 2 else 'vals'}"
            for i in range(num_of_inputs)
        ]
        decoder_input_names = decoder_inputs + pkv_input_names

        decoder_outputs = ['logits']
        pkv_output_names = [
            f"l{i // 4}_{'self_out' if i % 4 < 2 else 'cross'}_{'keys' if not i % 2 else 'vals'}"
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

        # With past key values Decoder convert
        torch.onnx.export(
            decoder_with_past,
            decoder_all_inputs,
            os.path.join(save_path, 'onnx', 'decoder_with_past.onnx'),
            export_params=True,
            do_constant_folding=False,
            opset_version=13,
            input_names=decoder_input_names,
            output_names=decoder_output_names,
            dynamic_axes=dyn_axis_params
        )

        # Initial decoder to produce past key values
        torch.onnx.export(
            decoder,
            (input_ids_dec, attention_mask_dec, enc_out),
            os.path.join(save_path, 'onnx', 'decoder.onnx'),
            export_params=True,
            do_constant_folding=False,
            opset_version=13,
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


if __name__ == "__main__":
    """
    Example:
    $ python convert_onnx.py --donut_model_path="naver-clova-ix/donut-base"or trained your model path  --onnx_save_path=result
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--donut_model_path', type=str, required=True)
    parser.add_argument('--onnx_save_path', type=str, default='result')
    args, left_argv = parser.parse_known_args()

    # Load Donut Model
    donut_model = DonutModel.from_pretrained(args.donut_model_path)
    if not os.path.exists(os.path.join(args.onnx_save_path, 'onnx')):
        os.makedirs(os.path.join(args.onnx_save_path, 'onnx'))

    # Save Vocab to json
    vocab = donut_model.decoder.tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}

    if not os.path.exists(os.path.join(args.onnx_save_path, 'onnx', 'vocab.json')):
        with open(os.path.join(args.onnx_save_path, 'onnx', 'vocab.json'), 'w', encoding='utf-8') as outfile:
            json.dump(vocab, outfile, ensure_ascii=False)
    else:
        os.remove(os.path.join(args.onnx_save_path, 'onnx', 'vocab.json'))
        with open(os.path.join(args.onnx_save_path, 'onnx', 'vocab.json'), 'w', encoding='utf-8') as outfile:
            json.dump(vocab, outfile, ensure_ascii=False)

    # Generate ONNX files
    generate_onnx(donut_model, save_path=args.onnx_save_path, config_path=args.donut_model_path)