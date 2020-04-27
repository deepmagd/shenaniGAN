from PIL import Image


def concate_horizontallly(img_1, img_2):
    res = Image.new('RGB', (img_1.width + img_2.width, img_1.height))
    res.paste(img_1, (0, 0))
    res.paste(img_2, (img_1.width, 0))
    return res
