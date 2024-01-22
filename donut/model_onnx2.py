import torch
from donut import DonutModel
import onnxruntime as ort
import numpy as np
import PIL
from PIL import ImageOps
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Donut 모델을 불러옵니다.
donut_model = DonutModel.from_pretrained('result/bscard3')

# DonutModel의 encoder와 decoder를 추출
encoder = donut_model.encoder
decoder = donut_model.decoder

# 임의의 입력 데이터 생성 (적절한 크기와 타입으로 변경해야 함)
dummy_image_tensor = torch.randn(1, 3, 960, 640) # encoder input 예시
dummy_input_ids = torch.randint(0, 100, (1,120))
dummy_attention_mask = torch.randint(0, 100, (1,120))
dummy_encoder_hidden_states = torch.randn(1,600, 1024)

# Encoder를 ONNX로 변환
torch.onnx.export(encoder,
                  dummy_image_tensor,
                  "result/encoder.onnx",
                  input_names=['input_image'],
                  output_names=['encoder_output'],
                  opset_version=13
                  )


# Decoder를 ONNX로 변환
torch.onnx.export(decoder,
                  (dummy_input_ids, dummy_attention_mask, dummy_encoder_hidden_states,),
                  "result/decoder.onnx",
                  input_names=['input_ids', 'attention_mask', 'encoder_hidden_states'],
                  output_names=['decoder_output'],
                  opset_version=13,
                  dynamic_axes={
                      'input_ids':{1:'sequence'},
                      'attention_mask':{1:'sequence'}}
                  )




def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ONNX 런타임 세션을 생성합니다.
encoder_sess = ort.InferenceSession('result/encoder.onnx')
decoder_sess = ort.InferenceSession('result/decoder.onnx')


# 입력 데이터를 준비합니다.
# 여기서는 임의의 텐서를 사용하였습니다.
tokenizer = decoder.tokenizer

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



test_image = PIL.Image.open('bscard.png')
encoder_input = prepare_input(img=test_image).unsqueeze(0).numpy()


# 인코더와 디코더 모델을 실행합니다.
encoder_output = encoder_sess.run(None, {'input_image': encoder_input})
#
#
# test_prompt = tokenizer('<s_2023_07_21>', add_special_tokens=False, return_tensors="pt")
# decoder_output_logits = decoder_sess.run(None,
#                                          {'input_ids': test_prompt['input_ids'].numpy(),
#                                           'attention_mask':test_prompt['attention_mask'].numpy(),
#                                           'encoder_hidden_states':encoder_output[0]})
#
#
# # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
# output_probs = softmax(decoder_output_logits[0])
#
# # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
# top_token = np.argmax(output_probs, axis=-1)
# print(top_token)
#
# # 생성된 토큰들을 문자열로 변환합니다.
# generated_text = decoder.tokenizer.decode(top_token[0][0])
# print(generated_text)


res = []
for i in range(120):
    if i == 0:
        prompt = '<s_2023_07_21>'
        print(f'Start Token: {prompt} \n')
        promt_ids = tokenizer(f'{prompt}', add_special_tokens=False, return_tensors="pt")
        decoder_output_logits = decoder_sess.run(None, 
                                         {'input_ids': promt_ids['input_ids'].numpy(), 
                                          'attention_mask':promt_ids['attention_mask'].numpy(), 
                                          'encoder_hidden_states':encoder_output[0]})
        # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
        output_probs = softmax(decoder_output_logits[0])

        # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
        top_token = np.argmax(output_probs, axis=-1)

        # 생성된 토큰들을 문자열로 변환합니다.
        generated_text = decoder.tokenizer.decode(top_token[0][0])
        res.append(generated_text)
        prompt = generated_text
    else:
        print(f'Generated Token: {prompt}')
        promt_ids = tokenizer(f'{prompt}', add_special_tokens=False, return_tensors="pt")


        decoder_output_logits = decoder_sess.run(None,
                                                 {'input_ids': promt_ids['input_ids'].numpy(),
                                                  'attention_mask': promt_ids['attention_mask'].numpy(),
                                                  'encoder_hidden_states': encoder_output[0]})

        # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
        output_probs = softmax(decoder_output_logits[0])

        # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
        top_token = np.argmax(output_probs, axis=-1)

        # 생성된 토큰들을 문자열로 변환합니다.
        generated_text = decoder.tokenizer.decode(top_token[0][0])
        res.append(generated_text)
        prompt = generated_text


print(res)
