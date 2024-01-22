from transformers import AutoTokenizer, MBartModel, MBartForCausalLM, XLMRobertaTokenizer
from transformers.file_utils import ModelOutput
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
import PIL
from PIL import ImageOps
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import onnxruntime as ort
import re


"""
Donut Model Test
"""

from donut import DonutModel
donut_model = DonutModel.from_pretrained('result/bscard3')

donut_encoder = donut_model.encoder
donut_decoder = donut_model.decoder
# print(donut_encoder)
# print()
# print(donut_decoder)
# print()

"""
Donut Encoder Test
"""

input_size = [960, 640]
align_long_axis = False
def prepare_input(img: PIL.Image.Image, random_padding: bool = False) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        img = img.convert("RGB")
        if align_long_axis and (
            (input_size[0] > input_size[1] and img.width > img.height)
            or (input_size[0] < input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(input_size))
        img.thumbnail((input_size[1], input_size[0]))
        delta_width = input_size[1] - img.width
        delta_height = input_size[0] - img.height
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
        return transforms.Compose(
             [
                  transforms.ToTensor(),
                  transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
             ]
        )(ImageOps.expand(img, padding))

# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=-1, keepdims=True)

test_image = PIL.Image.open('bscard.png')
test_image_tensor = prepare_input(img=test_image).unsqueeze(0).numpy()


onnx_session = ort.InferenceSession('result/donut_encoder.onnx')
encoder_output = onnx_session.run(['donutencoder_output'],
                                  {'input_img' : test_image_tensor})


donut_encoder_output = ModelOutput(last_hidden_state=encoder_output, attentions=None)
donut_encoder_output.last_hidden_state = torch.Tensor(donut_encoder_output.last_hidden_state)

tokenizer = donut_decoder.tokenizer

sftmax = torch.nn.Softmax(dim=1)

res = []
for i in range(120):
    if i == 0:
        prompt = '<s_2023_07_21>'
        print(f'Start Token: {prompt} \n')
        promt_ids = tokenizer(f'{prompt}', add_special_tokens=False, return_tensors="pt")
        input_ids = torch.Tensor([[0, promt_ids.input_ids[0, 0]]]).type(torch.int64)
        attention_mask = torch.Tensor([[1 for j in range(i+2)]]).type(torch.int64)
        decoder_output_logits = donut_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=donut_encoder_output.last_hidden_state
        )
        # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
        lo = decoder_output_logits.logits.squeeze(0)
        output_probs = sftmax(decoder_output_logits.logits.squeeze(0))

        # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
        top_token = torch.argmax(output_probs)

        # 생성된 토큰들을 문자열로 변환합니다.
        generated_text = donut_decoder.tokenizer.decode(top_token.item())
        res.append(generated_text)
        prompt = generated_text
    else:
        print(f'Generated Token: {prompt}')
        promt_ids = tokenizer(f'{prompt}', add_special_tokens=False, return_tensors="pt")
        attention_mask = torch.Tensor([int(1) for i in range((1+i))]).unsqueeze(0)
        decoder_output_logits = donut_decoder(
            input_ids=promt_ids['input_ids'],
            attention_mask=attention_mask,
            encoder_hidden_states=donut_encoder_output.last_hidden_state
        )
        # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
        lo = decoder_output_logits.logits.squeeze(0)
        output_probs = sftmax(decoder_output_logits.logits.squeeze(0))

        # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
        top_token = torch.argmax(output_probs)

        # 생성된 토큰들을 문자열로 변환합니다.
        generated_text = donut_decoder.tokenizer.decode(top_token.item())
        res.append(generated_text)
        prompt = generated_text
