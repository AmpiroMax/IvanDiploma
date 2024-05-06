import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional


def create_mask_from_image(image: Image) -> np.ndarray:
    array = np.array(image)
    h, w, d = array.shape
    mask = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            pix = array[i, j]
            if pix[3] < 250:  # Пустой
                mask[i, j] = 0
            elif pix[0] == pix[1] == pix[2] == 255:  # Белый
                mask[i, j] = 1
            elif pix[0] == pix[1] == pix[2] == 0:  # Черный
                mask[i, j] = 2
            elif 250 <= pix[0] <= 255:  # Красный
                mask[i, j] = 3
            elif 250 <= pix[1] <= 255:  # Зеленый
                mask[i, j] = 4
            elif 250 <= pix[2] <= 255:  # Синий
                mask[i, j] = 5

    return mask


def create_image_from_mask(mask: np.array) -> Image:
    h, w = mask.shape
    image = np.zeros((h, w, 4), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            pix_class = mask[i, j]
            match pix_class:
                case 0:
                    image[i, j] = (0, 0, 0, 0)
                case 1:
                    image[i, j] = (255, 255, 255, 255)
                case 2:
                    image[i, j] = (0, 0, 0, 255)
                case 3:
                    image[i, j] = (255, 0, 0, 255)
                case 4:
                    image[i, j] = (0, 255, 0, 255)
                case 5:
                    image[i, j] = (0, 0, 255, 255)

    image = Image.fromarray(image)
    return image


def test_on_image(model: torch.nn.Module, img_data: Optional[str | Image.Image]) -> Image:

    image = None
    if isinstance(img_data, str):
        image = Image.open(img_data)
    elif isinstance(img_data, Image.Image):
        image = img_data
    else:
        raise ValueError(
            "Не то передали в качестве переменной img_data. Ожидается str или PIL.Image")

    image_to_tensor = transforms.Compose([
        transforms.PILToTensor(),
    ])

    image_tensor = image_to_tensor(image).type(torch.FloatTensor)[None, ...]

    with torch.no_grad():
        predict = model.to("cuda")(image_tensor.to("cuda"))

    label = create_image_from_mask(
        torch.argmax(predict, dim=1)[0].cpu().numpy())
    return label
