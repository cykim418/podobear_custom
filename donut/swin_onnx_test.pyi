from transformers import AutoTokenizer, MBartModel, MBartForCausalLM, XLMRobertaTokenizer
from transformers.file_utils import ModelOutput
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
import PIL
from PIL import ImageOps
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import re
import onnxruntime


"""
Donut Model Test
"""

from donut import DonutModel
donut_model = DonutModel.from_pretrained('result/bscard3')

donut_encoder = donut_model.encoder
donut_decoder = donut_model.decoder


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



test_image = PIL.Image.open('bscard.png')
test_image_tensor = prepare_input(img=test_image).unsqueeze(0)
print("Test Image shape: ",test_image_tensor.shape)

# donut
# encoder_output = donut_encoder(test_image_tensor)

# onnx

test_image_tensor = test_image_tensor.numpy()
# print(test_image_tensor.shape)
onnx_session = onnxruntime.InferenceSession('result/donut_encoder.onnx')
encoder_output = onnx_session.run(['donutencoder_output'],
                                  {'input_img' : test_image_tensor})


donut_encoder_output = ModelOutput(last_hidden_state=encoder_output, attentions=None)
donut_encoder_output.last_hidden_state = torch.Tensor(donut_encoder_output.last_hidden_state)

# output_tensor = torch.Tensor(donut_encoder_output.last_hidden_state[0])
# donut_encoder_output = output_tensor.squeeze(0)
# print(output_tensor.shape)
print('-------------------------------------------------------')
print()




donut_tokenizer = donut_decoder.tokenizer
test_input = donut_tokenizer('<s_2023_07_21>', add_special_tokens=False, return_tensors="pt")
print('Test input text: <s_2023_07_21>.')
print('Test input text to tensor: ', test_input)
print("input_ids shape: ",test_input['input_ids'].shape, "attention_mask shape: ", test_input['attention_mask'].shape)
print('-------------------------------------------------------')


donut_decoder_output = donut_decoder.model.generate(
    decoder_input_ids=test_input['input_ids'],
    encoder_outputs=donut_encoder_output,
    max_length=120,
    early_stopping=True,
    pad_token_id=donut_tokenizer.pad_token_id,
    eos_token_id=donut_tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[donut_tokenizer.unk_token_id]],
    return_dict_in_generate=True,
    output_attentions=False,
)

output = {"predictions": list()}
return_json = False
for seq in donut_tokenizer.batch_decode(donut_decoder_output.sequences):
    seq = seq.replace(donut_tokenizer.eos_token, "").replace(donut_tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
    output["predictions"].append(seq)

print(output)
