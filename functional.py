import math, torch, torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageOps


def solarize_add(img, addition, threshold):
    img = F.pil_to_tensor(img)
    added_img = img + torch.tensor(addition, dtype=img.dtype, device=img.device)
    added_img = torch.clamp(added_img, 0, 255)
    bound = torch.tensor(255, dtype=img.dtype, device=img.device)
    inverted_img = bound - added_img
    return F.to_pil_image(torch.where(added_img >= threshold, inverted_img, added_img))


def color(img, magnitude):
    return ImageEnhance.Color(img).enhance(magnitude)


def contrast(img, magnitude):
    return ImageEnhance.Contrast(img).enhance(magnitude)


def brightness(img, magnitude):
    return ImageEnhance.Brightness(img).enhance(magnitude)


def sharpness(img, magnitude):
    return ImageEnhance.Sharpness(img).enhance(magnitude)


def cutout(img, pad_size, replace):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    center_h, center_w = torch.randint(high=h, size=(1,)), torch.randint(high=w, size=(1,))
    low_h, high_h = torch.clamp(center_h-pad_size, 0, h).item(), torch.clamp(center_h+pad_size, 0, h).item()
    low_w, high_w = torch.clamp(center_w-pad_size, 0, w).item(), torch.clamp(center_w+pad_size, 0, w).item()
    cutout_img = img.clone()
    cutout_img[:, low_h:high_h, low_w:high_w] = replace
    return F.to_pil_image(cutout_img)


def bbox_cutout(img, bboxs, pad_fraction, replace_with_mean):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    random_index = torch.randint(bboxs.size(0), size=(1,)).item()
    chosen_bbox = bboxs[random_index]
    min_x, min_y, max_x, max_y = chosen_bbox
    min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
    
    if (min_x == max_x) or (min_y == max_y): return F.to_pil_image(img)
    
    mask_x, mask_y = torch.randint(low=min_x, high=max_x, size=(1,)), torch.randint(low=min_y, high=max_y, size=(1,))
    mask_w, mask_h = pad_fraction * w / 2, pad_fraction * h / 2
    
    x_min, x_max = int(torch.clamp(mask_x-mask_w, 0, w).item()), int(torch.clamp(mask_x+mask_w, 0, w).item())
    y_min, y_max = int(torch.clamp(mask_y-mask_h, 0, h).item()), int(torch.clamp(mask_y+mask_h, 0, h).item())
    
    if replace_with_mean == True: replace = torch.mean(img[:, min_y:max_y, min_x:max_x]).item()
    else: replace = 128
    
    cutout_img = img.clone()
    cutout_img[:, y_min:y_max, x_min:x_max] = replace
    return F.to_pil_image(cutout_img)


def _rotate_bbox(img, bboxs, degrees):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    
    rotate_bboxs = []
    rotate_matrix = torch.FloatTensor([[math.cos(degrees*math.pi/180), math.sin(degrees*math.pi/180)], 
                                       [-math.sin(degrees*math.pi/180), math.cos(degrees*math.pi/180)]])
    for bbox in bboxs:
        min_x, min_y, max_x, max_y = bbox
        rel_min_x, rel_max_x, rel_min_y, rel_max_y = min_x-w/2, max_x-w/2, min_y-h/2, max_y-h/2
        coords = torch.FloatTensor([[rel_min_x, rel_min_y], 
                                    [rel_min_x, rel_max_y], 
                                    [rel_max_x, rel_max_y], 
                                    [rel_max_x, rel_min_y]])
        rotate_coords = torch.matmul(rotate_matrix, coords.t()).t()
        x_min, y_min = torch.min(rotate_coords, dim=0)[0]
        x_max, y_max = torch.max(rotate_coords, dim=0)[0]
        
        rotate_min_x, rotate_max_x = torch.clamp(x_min+w/2, 0, w),torch.clamp(x_max+w/2, 0, w)
        rotate_min_y, rotate_max_y = torch.clamp(y_min+h/2, 0, h),torch.clamp(y_max+h/2, 0, h)
        rotate_bboxs.append(torch.FloatTensor([rotate_min_x, rotate_min_y, rotate_max_x, rotate_max_y]))
    return torch.stack(rotate_bboxs)


def translate_bbox(img, bboxs, pixels, replace, shift_horizontal):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    
    translate_bboxs = []
    if shift_horizontal:
        for bbox in bboxs:
            min_x, min_y, max_x, max_y = bbox
            translate_min_x, translate_max_x = torch.clamp(min_x+pixels, 0, w), torch.clamp(max_x+pixels, 0, w)
            translate_min_x, translate_max_x = int(translate_min_x.item()), int(translate_max_x.item())
            translate_bboxs.append(torch.FloatTensor([translate_min_x, min_y, translate_max_x, max_y]))
    else:
        for bbox in bboxs:
            min_x, min_y, max_x, max_y = bbox
            translate_min_y, translate_max_y = torch.clamp(min_y+pixels, 0, h), torch.clamp(max_y+pixels, 0, h)
            translate_min_y, translate_max_y = int(translate_min_y.item()), int(translate_max_y.item())
            translate_bboxs.append(torch.FloatTensor([min_x, translate_min_y, max_x, translate_max_y]))
    return torch.stack(translate_bboxs)


def shear_with_bboxes(img, bboxs, level, replace, shift_horizontal):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    
    shear_bboxs = []
    if shift_horizontal:
        shear_matrix = torch.FloatTensor([[1, -level], 
                                          [0, 1]])
        for bbox in bboxs:
            min_x, min_y, max_x, max_y = bbox
            coords = torch.FloatTensor([[min_x, min_y], 
                                        [min_x, max_y], 
                                        [max_x, max_y], 
                                        [max_x, min_y]])
            shear_coords = torch.matmul(shear_matrix, coords.t()).t()
            x_min, y_min = torch.min(shear_coords, dim=0)[0]
            x_max, y_max = torch.max(shear_coords, dim=0)[0]
            shear_min_x, shear_max_x = torch.clamp(x_min, 0, w), torch.clamp(x_max, 0, w)
            shear_min_y, shear_max_y = torch.clamp(y_min, 0, h), torch.clamp(y_max, 0, h)
            shear_bboxs.append(torch.FloatTensor([shear_min_x, shear_min_y, shear_max_x, shear_max_y]))
    else:
        shear_matrix = torch.FloatTensor([[1, 0], 
                                          [-level, 1]])
        for bbox in bboxs:
            min_x, min_y, max_x, max_y = bbox
            coords = torch.FloatTensor([[min_x, min_y], 
                                        [min_x, max_y], 
                                        [max_x, max_y], 
                                        [max_x, min_y]])
            shear_coords = torch.matmul(shear_matrix, coords.t()).t()
            x_min, y_min = torch.min(shear_coords, dim=0)[0]
            x_max, y_max = torch.max(shear_coords, dim=0)[0]
            shear_min_x, shear_max_x = torch.clamp(x_min, 0, w), torch.clamp(x_max, 0, w)
            shear_min_y, shear_max_y = torch.clamp(y_min, 0, h), torch.clamp(y_max, 0, h)
            shear_bboxs.append(torch.FloatTensor([shear_min_x, shear_min_y, shear_max_x, shear_max_y]))
    return torch.stack(shear_bboxs)


def rotate_only_bboxes(img, bboxs, p, degrees, replace):
    img = F.pil_to_tensor(img)
    rotate_img = torch.zeros_like(img)

    for bbox in bboxs:
        if torch.rand(1) < p:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            bbox_rotate_img = F.to_pil_image(img[:, min_y:max_y+1, min_x:max_x+1]).rotate(degrees, fillcolor=(replace,replace,replace))
            rotate_img[:, min_y:max_y+1, min_x:max_x+1] = F.pil_to_tensor(bbox_rotate_img)
    return F.to_pil_image(torch.where(rotate_img != 0, rotate_img, img))


def shear_only_bboxes(img, bboxs, p, level, replace, shift_horizontal):
    img = F.pil_to_tensor(img)
    shear_img = torch.zeros_like(img)
    
    for bbox in bboxs:
        if torch.rand(1) < p:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            
            bbox_shear_img = F.to_pil_image(img[:, min_y:max_y+1, min_x:max_x+1])
            if shift_horizontal:
                bbox_shear_img = bbox_shear_img.transform(bbox_shear_img.size, Image.AFFINE, (1,level,0,0,1,0), fillcolor=(replace,replace,replace))
            else:
                bbox_shear_img = bbox_shear_img.transform(bbox_shear_img.size, Image.AFFINE, (1,0,0,level,1,0), fillcolor=(replace,replace,replace))
            shear_img[:, min_y:max_y+1, min_x:max_x+1] = F.pil_to_tensor(bbox_shear_img)

    return F.to_pil_image(torch.where(shear_img != 0, shear_img, img))


def translate_only_bboxes(img, bboxs, p, pixels, replace, shift_horizontal):
    img = F.pil_to_tensor(img)
    translate_img = torch.zeros_like(img)
    
    for bbox in bboxs:
        if torch.rand(1) < p:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            
            bbox_tran_img = F.to_pil_image(img[:, min_y:max_y+1, min_x:max_x+1])
            if shift_horizontal:
                bbox_tran_img = bbox_tran_img.transform(bbox_tran_img.size, Image.AFFINE, (1,0,-pixels,0,1,0), fillcolor=(replace,replace,replace))
            else:
                bbox_tran_img = bbox_tran_img.transform(bbox_tran_img.size, Image.AFFINE, (1,0,0,0,1,-pixels), fillcolor=(replace,replace,replace))
            translate_img[:, min_y:max_y+1, min_x:max_x+1] = F.pil_to_tensor(bbox_tran_img)
    
    return F.to_pil_image(torch.where(translate_img != 0, translate_img, img))


def flip_only_bboxes(img, bboxs, p):
    img = F.pil_to_tensor(img)
    flip_img = torch.zeros_like(img)
    
    for bbox in bboxs:
        if torch.rand(1) < p:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            flip_img[:, min_y:max_y+1, min_x:max_x+1] = F.hflip(img[:, min_y:max_y+1, min_x:max_x+1])
    
    return F.to_pil_image(torch.where(flip_img != 0, flip_img, img))


def solarize_only_bboxes(img, bboxs, p, threshold):
    img = F.pil_to_tensor(img)
    for bbox in bboxs:
        if torch.rand(1) < p:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            solarize_img = img[:, min_y:max_y+1, min_x:max_x+1]
            solarize_img = F.to_pil_image(solarize_img)
            solarize_img = ImageOps.solarize(solarize_img, threshold=threshold)
            solarize_img = F.pil_to_tensor(solarize_img)
            img[:, min_y:max_y+1, min_x:max_x+1] = solarize_img
    return F.to_pil_image(img)


def equalize_only_bboxes(img, bboxs, p):
    img = F.pil_to_tensor(img)
    for bbox in bboxs:
        if torch.rand(1) < p:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            equalize_img = img[:, min_y:max_y+1, min_x:max_x+1]
            equalize_img = F.to_pil_image(equalize_img)
            equalize_img = ImageOps.equalize(equalize_img)
            equalize_img = F.pil_to_tensor(equalize_img)
            img[:, min_y:max_y+1, min_x:max_x+1] = equalize_img
    return F.to_pil_image(img)


def cutout_only_bboxes(img, bboxs, p, pad_size, replace):
    img = F.pil_to_tensor(img)
    cutout_img = img.clone()
    
    for bbox in bboxs:
        if torch.rand(1) < p:
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x.item()), int(min_y.item()), int(max_x.item()), int(max_y.item())
            
            cutout_x, cutout_y = torch.randint(low=min_x, high=max_x, size=(1,)), torch.randint(low=min_y, high=max_y, size=(1,))
            
            y_min, y_max = int(torch.clamp(cutout_y-pad_size, min_y, max_y).item()), int(torch.clamp(cutout_y+pad_size, min_y, max_y).item())
            x_min, x_max = int(torch.clamp(cutout_x-pad_size, min_x, max_x).item()), int(torch.clamp(cutout_x+pad_size, min_x, max_x).item())
            
            cutout_img[:, y_min:y_max, x_min:x_max] = replace

    return F.to_pil_image(cutout_img)