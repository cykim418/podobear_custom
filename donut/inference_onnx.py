import os
import re
import json
import operator
import functools
import argparse

import PIL
from PIL import ImageOps
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import onnxruntime as ort


# Functions
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


def to_tensor(img):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])(img)


def prepare_input(img, config, random_padding=False):
    img = img.convert("RGB")
    if config['align_long_axis'] and (
            (config['input_size'][0] > config['input_size'][1] and img.width > img.height)
            or (config['input_size'][0] < config['input_size'][1] and img.width < img.height)
    ):
        img = rotate(img, angle=-90, expand=True)
    img = resize(img, min(config['input_size']))
    img.thumbnail((config['input_size'][1], config['input_size'][0]))
    delta_width = config['input_size'][1] - img.width
    delta_height = config['input_size'][0] - img.height
    if random_padding:
        pad_width = np.random.randint(low=0, high=delta_width + 1)
        pad_height = np.random.randint(low=0, high=delta_height + 1)
    else:
        pad_width = delta_width // 2
        pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    return to_tensor(ImageOps.expand(img, padding))


def token2json(tokens, donut_model_path, is_inner_value=False):
    with open(os.path.join(donut_model_path, 'added_tokens.json'), 'r', ) as f:
        add_tokens_json = json.load(f)

    output = dict()

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(content, is_inner_value=True)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if (
                                leaf in add_tokens_json
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                        ):
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokens[6:], is_inner_value=True)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}


# Inference ONNX models
def inference_onnx(path, donut_model_path, image_path, prompt):
    """
    Inference onnx file
    """
    # Load ONNX files
    if 'onnx_quantized' in path:
        encoder_sess = ort.InferenceSession(os.path.join(path, 'quantized_encoder.ort'))
        decoder_sess1 = ort.InferenceSession(os.path.join(path, 'quantized_decoder.ort'))
        decoder_sess2 = ort.InferenceSession(os.path.join(path, 'quantized_decoder_with_past.ort'))
    else:
        encoder_sess = ort.InferenceSession(os.path.join(path, 'encoder.onnx'))
        decoder_sess1 = ort.InferenceSession(os.path.join(path, 'decoder.onnx'))
        decoder_sess2 = ort.InferenceSession(os.path.join(path, 'decoder_with_past.onnx'))

    # Load json files
    if 'onnx_quantized' in path:
        with open(os.path.join(os.path.dirname(path), 'vocab.json'), 'r', encoding='utf-8') as t:
            tokenizer = json.load(t)
    else:
        with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as t:
            tokenizer = json.load(t)

    with open(os.path.join(donut_model_path, 'added_tokens.json'), 'r', encoding='utf-8') as f:
        add_tokens_json = json.load(f)

    with open(os.path.join(donut_model_path, 'config.json'), 'r') as c:
        config = json.load(c)

    test_image = PIL.Image.open(image_path)
    encoder_input = prepare_input(img=test_image, config=config).unsqueeze(0).numpy()
    encoder_output = encoder_sess.run(None, {'image': encoder_input})

    start_prompt = f'<s_{prompt}>'
    start_prompt_ids = np.array([[add_tokens_json[start_prompt]]]).astype(np.int32)
    start_prompt_attention_mask = np.array([[1]]).astype(np.int32)

    res = []
    for i in range(config['max_length']):
        if i == 0:
            # res.append(start_prompt)
            decoder_output_logits1 = decoder_sess1.run(None,
                                                       {'input_ids': start_prompt_ids,
                                                        'attention_mask': start_prompt_attention_mask,
                                                        'encoder_hidden_states': encoder_output[0]})

            # create past key values
            list_pkv = tuple(torch.from_numpy(x) for x in decoder_output_logits1[1:])
            out_past_key_values = tuple(
                list_pkv[i: i + 4] for i in range(0, len(list_pkv), 4)
            )

            output_probs = softmax(decoder_output_logits1[0][0])
            top_token = np.argmax(output_probs, axis=-1)

            generated_text = tokenizer[str(top_token[0])]
            res.append(generated_text)

        else:
            attention_mask = np.array([[1 for _ in range(i + 1)]]).astype(np.int32)
            decoder_inputs = {
                "input_ids": np.array([[top_token[0]]]),
                "attention_mask": attention_mask,
            }

            flat_past_key_values = functools.reduce(operator.iconcat, out_past_key_values, [])

            input_names = [x.name for x in decoder_sess2.get_inputs()]
            inputs = [
                         np.array([[top_token[0]]]).astype(np.int32),
                         np.array(attention_mask).astype(np.int32)] + [tensor.numpy() for tensor in
                                                                       flat_past_key_values]
            decoder_inputs = dict(zip(input_names, inputs))

            decoder_output_logits2 = decoder_sess2.run(None, decoder_inputs)

            list_pkv = tuple(torch.from_numpy(x) for x in decoder_output_logits2[1:])
            out_past_key_values = tuple(
                list_pkv[i: i + 4] for i in range(0, len(list_pkv), 4)
            )

            output_probs = softmax(decoder_output_logits2[0][0])
            top_token = np.argmax(output_probs, axis=-1)ttttttttttttttttttttt

            generated_text = tokenizer[str(top_token[0])]
            res.append(generated_text)

    predict = ""
    for t in res:
        t = t.replace('▁', ' ')
        if t == "<unk>":
            continue
        predict += t
    predict = token2json(tokens=predict, donut_model_path=donut_model_path)
    return res, predict


if __name__ == "__main__":
    """
    Example:
    $ python inference_onnx.py --donut_model_path="naver-clova-ix/donut-base"or trained your model path  --image_path="image path" --prompt="task name"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_save_path', type=str, default='result/onnx/onnx_quantized')
    parser.add_argument('--donut_model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    args, left_argv = parser.parse_known_args()

    res, predict = inference_onnx(path=args.onnx_save_path, donut_model_path=args.donut_model_path,
                                  image_path=args.image_path, prompt=args.prompt)
    print(predict)