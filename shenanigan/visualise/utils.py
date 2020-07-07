from PIL import Image


def concate_horizontallly(real_img: Image, **kwargs) -> Image:
    stage1_img = kwargs.get('stage1_img')
    stage2_img = kwargs.get('stage2_img')

    if stage1_img is None and stage2_img is None:
        raise Exception('Please provide `concate_horizontallly` with either a stage1_img or a stage2_img')

    if stage1_img is not None and stage2_img is None:
        return concate_two_images_horizontallly(real_img, stage1_img)

    if stage1_img is None and stage2_img is not None:
        return concate_two_images_horizontallly(real_img, stage2_img)

    if stage1_img is not None and stage2_img is not None:
        stage1_img_resized = stage1_img.resize((stage2_img.width, stage2_img.height))
        res = concate_two_images_horizontallly(real_img, stage1_img_resized)
        res2 = concate_two_images_horizontallly(res, stage2_img)
        return res2

def concate_two_images_horizontallly(img_1: Image, img_2: Image) -> Image:
    res = Image.new('RGB', (img_1.width + img_2.width, img_1.height))
    res.paste(img_1, (0, 0))
    res.paste(img_2, (img_1.width, 0))
    return res
