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
dummy_input_ids = torch.randint(0, 100, (1,1))
dummy_attention_mask = torch.randint(1, (1,1))
dummy_encoder_hidden_states = torch.randn(1,600, 1024)


# Encoder를 ONNX로 변환
torch.onnx.export(encoder,
                  dummy_image_tensor,
                  "result/encoder.onnx",
                  input_names=['input_image'],
                  output_names=['encoder_output'],
                  opset_version=13
                  )



# Decoder without past_key_values를 ONNX로 변환
torch.onnx.export(decoder,
                  (dummy_input_ids, dummy_attention_mask, dummy_encoder_hidden_states,),
                  "result/decoder_without_past.onnx",
                  input_names=['input_ids', 'attention_mask', 'encoder_hidden_states'],
                  output_names=['decoder_output'],
                  opset_version=13,
                  dynamic_axes={
                      'input_ids':{1:'sequence'},
                      'attention_mask':{1:'sequence'}}
                  )


dummy_image_tensor = torch.randn(1, 3, 960, 640) # encoder input 예시
dummy_input_ids = torch.randint(0, 100, (1,1))
dummy_attention_mask = torch.randint(1, (1,2))
dummy_encoder_hidden_states = torch.randn(1,600, 1024)
dummy_past_key_values = (
    (torch.randn(1,16,1,64), torch.randn(1,16,1,64), torch.randn(1,16,600,64), torch.randn(1,16,600,64)),
    (torch.randn(1,16,1,64), torch.randn(1,16,1,64), torch.randn(1,16,600,64), torch.randn(1,16,600,64)),
    (torch.randn(1,16,1,64), torch.randn(1,16,1,64), torch.randn(1,16,600,64), torch.randn(1,16,600,64)),
    (torch.randn(1,16,1,64), torch.randn(1,16,1,64), torch.randn(1,16,600,64), torch.randn(1,16,600,64))
)


# Decoder를 ONNX로 변환
torch.onnx.export(decoder,
                  (dummy_input_ids, dummy_attention_mask, dummy_encoder_hidden_states, dummy_past_key_values,),
                  "result/decoder_with_past.onnx",
                  input_names=['input_ids', 'attention_mask', 'encoder_hidden_states', 'past_key_values'],
                  output_names=['decoder_output2'],
                  opset_version=13,
                  dynamic_axes={                  
                      'attention_mask':{1:'sequence'}}
                  )




def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

# ONNX 런타임 세션을 생성합니다.
encoder_sess = ort.InferenceSession('result/encoder.onnx')
decoder_sess1 = ort.InferenceSession('result/decoder_without_past.onnx')
decoder_sess2 = ort.InferenceSession('result/decoder_with_past.onnx')


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





prompt = '<s_2023_07_21>'
print(f'Start Token: {prompt}')
promt_ids = tokenizer(f'{prompt}', add_special_tokens=False, return_tensors="pt")


res = []
past_key_values = []

for i in range(20):
    if i == 0:
        res.append(prompt)

        decoder_output_logits1 = decoder_sess1.run(None, 
                                            {'input_ids': promt_ids['input_ids'].numpy(), 
                                            'attention_mask':promt_ids['attention_mask'].numpy(), 
                                            'encoder_hidden_states':encoder_output[0]})
        tmp = []
        for i in range(1, len(decoder_output_logits1)):            
            if i % 4 != 0:
                tmp.append(decoder_output_logits1[i])
            else:
                tmp.append(decoder_output_logits1[i])
                tmp = tuple(tmp)
                past_key_values.append(tmp)
                tmp = []
                
        past_key_values = tuple(past_key_values)
        print(past_key_values)

        # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
        output_probs = softmax(decoder_output_logits1[0])
        

        # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
        top_token = np.argmax(output_probs, axis=-1)

        # 생성된 토큰들을 문자열로 변환합니다.
        generated_text = tokenizer.decode(top_token[0][0])
        res.append(generated_text)
        prompt += generated_text
        print(f'Generated Token: {generated_text}')
    
    else:
        attention = np.array([[1 for _ in range(i+1)]])
        print("attention Mask -> ",attention)
        print('top token', top_token)
        

        decoder_output_logits2 = decoder_sess2.run(None, 
                                            {'input_ids': top_token.astype(np.int64), 
                                            'attention_mask':attention.astype(np.int64), 
                                            'encoder_hidden_states':encoder_output[0],
                                            'past_key_values':past_key_values})
        past_key_values = []
        tmp = []
        for i in range(1, len(decoder_output_logits2)):            
            if i % 4 != 0:
                tmp.append(decoder_output_logits2[i])
            else:
                tmp.append(decoder_output_logits2[i])
                tmp = tuple(tmp)
                past_key_values.append(tmp)
                tmp = []
                
        past_key_values = tuple(past_key_values)


        # softmax 함수를 적용하여 로짓을 확률로 변환합니다.
        output_probs = softmax(decoder_output_logits2[0])

        # 가장 확률이 높은 토큰의 인덱스를 찾습니다.
        top_token = np.argmax(output_probs, axis=-1)

        # 생성된 토큰들을 문자열로 변환합니다.
        generated_text = tokenizer.decode(top_token[0][0])
        res.append(generated_text)
        prompt += generated_text
        print(f'Generated Token: {generated_text}')
    
print(res)
