"""
Coior Augmentation: Autocontrast, Brightness, Color, Contrast, Equalize, Posterize, Sharpness, Solarize, SolarizeAdd

Geometric Augmentation: Rotate_BBox, ShearX_BBox, ShearY_BBox, TranslateX_BBox, TranslateY_BBox, Flip

Mask Augmentation: Cutout

Color Augmentation based on BBoxes: Equalize_Only_BBoxes, Solarize_Only_BBoxes

Geometric Augmentation based on BBoxes: Rotate_Only_BBoxes, ShearX_Only_BBoxes, ShearY_Only_BBoxes, 
                                        TranslateX_Only_BBoxes, TranslateY_Only_BBoxes, Flip_Only_BBoxes

Mask Augmentation based on BBoxes: BBox_Cutout, Cutout_Only_BBoxes

"""


import torch, torchvision, functional
import torchvision.transforms.functional as F

from PIL import Image, ImageOps


### Basic Augmentation
class Compose:
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxs):
        for t in self.transforms:
            image, bboxs = t(image, bboxs)
        return image, bboxs


class ToTensor:
    """
    Converts a PIL Image or numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    Only applied to image, not bboxes.
    """
    def __call__(self, image, bboxs):
        return F.to_tensor(image), bboxs
    
    
class Normalize(torch.nn.Module):
    """
    Normalize a tensor image with mean and standard deviation.
    Only applied to image, not bboxes.
    """
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, image, bboxs):
        return F.normalize(image, self.mean, self.std, self.inplace), bboxs


### Coior Augmentation
class AutoContrast(torch.nn.Module):
    """
    Autocontrast the pixels of the given image.
    Only applied to image, not bboxes.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            autocontrast_image = ImageOps.autocontrast(image)
            return autocontrast_image, bboxs
        else:
            return image, bboxs
    
    
class Brightness(torch.nn.Module):
    """
    Adjust image brightness using magnitude.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, magnitude):
        super().__init__()
        self.p = p
        self.magnitude = magnitude

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            brightness_image = functional.brightness(image, self.magnitude)
            return brightness_image, bboxs
        else:
            return image, bboxs
        

class Color(torch.nn.Module):
    """
    Adjust image color balance using magnitude.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, magnitude):
        super().__init__()
        self.p = p
        self.magnitude = magnitude

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            color_image = functional.color(image, self.magnitude)
            return color_image, bboxs
        else:
            return image, bboxs
    
    
class Contrast(torch.nn.Module):
    """
    Adjust image contrast using magnitude.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, magnitude):
        super().__init__()
        self.p = p
        self.magnitude = magnitude

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            contrast_image = functional.contrast(image, self.magnitude)
            return contrast_image, bboxs
        else:
            return image, bboxs

        
class Equalize(torch.nn.Module):
    """
    Equalize the histogram of the given image.
    Only applied to image, not bboxes.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            equalize_image = ImageOps.equalize(image)
            return equalize_image, bboxs
        else:
            return image, bboxs
    
    
class Posterize(torch.nn.Module):
    """
    Posterize the image by reducing the number of bits for each color channel.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, bits):
        super().__init__()
        self.p = p
        self.bits = int(bits)

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            posterize_image = ImageOps.posterize(image, self.bits)
            return posterize_image, bboxs
        else:
            return image, bboxs
    

class Sharpness(torch.nn.Module):
    """
    Adjust image sharpness using magnitude.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, magnitude):
        super().__init__()
        self.p = p
        self.magnitude = magnitude

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            sharpness_image = functional.sharpness(image, self.magnitude)
            return sharpness_image, bboxs
        else:
            return image, bboxs
    
    
class Solarize(torch.nn.Module):
    """
    Solarize the image by inverting all pixel values above a threshold.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, threshold):
        super().__init__()
        self.p = p
        self.threshold = threshold

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            solarize_image = ImageOps.solarize(image, self.threshold)
            return solarize_image, bboxs
        else:
            return image, bboxs
    
    
class SolarizeAdd(torch.nn.Module):
    """
    Solarize the image by inverting all pixel values above a threshold.
    Add addition amount to image and then clip the pixel value to 0~255 or 0~1.
    Parameter addition must be integer.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, addition, threshold=128):
        super().__init__()
        self.p = p
        self.addition = addition
        self.threshold = threshold

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            solarize_add_image = functional.solarize_add(image, self.addition, self.threshold)
            return solarize_add_image, bboxs
        else:
            return image, bboxs
    
    
### Geometric Augmentation
class Rotate_BBox(torch.nn.Module):
    """
    Rotate image by degrees and change bboxes according to rotated image.
    The pixel values filled in will be of the value replace.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p, degrees, replace=128):
        super().__init__()
        self.p = p
        self.degrees = degrees
        self.replace = replace

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            rotate_image = image.rotate(self.degrees, fillcolor=(self.replace, self.replace, self.replace))
            if bboxs == None:
                return rotate_image, bboxs
            else:
                rotate_bbox = functional._rotate_bbox(image, bboxs, self.degrees)
                return rotate_image, rotate_bbox
        else:
            return image, bboxs
        
        
class ShearX_BBox(torch.nn.Module):
    """
    Shear image and change bboxes on X-axis.
    The pixel values filled in will be of the value replace.
    Level is usually between -0.3~0.3.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p, level, replace=128):
        super().__init__()
        self.p = p
        self.level = level
        self.replace = replace

    def forward(self, image, bboxs):
        if (torch.rand(1) < self.p) and (bboxs != None):
            shear_image = image.transform(image.size, Image.AFFINE, (1, self.level, 0, 0, 1, 0), fillcolor=(self.replace, self.replace, self.replace))
            if bboxs == None:
                return shear_image, bboxs
            else:
                shear_bbox = functional.shear_with_bboxes(image, bboxs, self.level, self.replace, shift_horizontal=True)
                return shear_image, shear_bbox
        else:
            return image, bboxs
        
        
class ShearY_BBox(torch.nn.Module):
    """
    Shear image and change bboxes on Y-axis.
    The pixel values filled in will be of the value replace.
    Level is usually between -0.3~0.3.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p, level, replace=128):
        super().__init__()
        self.p = p
        self.level = level
        self.replace = replace

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            shear_image = image.transform(image.size, Image.AFFINE, (1, 0, 0, self.level, 1, 0), fillcolor=(self.replace, self.replace, self.replace))
            if bboxs == None:
                return shear_image, bboxs
            else:
                shear_bbox = functional.shear_with_bboxes(image, bboxs, self.level, self.replace, shift_horizontal=False)
                return shear_image, shear_bbox
        else:
            return image, bboxs
        
        
class TranslateX_BBox(torch.nn.Module):
    """
    Translate image and bboxes on X-axis.
    The pixel values filled in will be of the value replace.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p, pixels, replace=128):
        super().__init__()
        self.p = p
        self.pixels = pixels
        self.replace = replace

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            translate_image = image.transform(image.size, Image.AFFINE, (1, 0, -self.pixels, 0, 1, 0), fillcolor=(self.replace, self.replace, self.replace))
            if bboxs == None:
                return translate_image, bboxs
            else:
                translate_bbox = functional.translate_bbox(image, bboxs, self.pixels, self.replace, shift_horizontal=True)
                return translate_image, translate_bbox
        else:
            return image, bboxs
    
    
class TranslateY_BBox(torch.nn.Module):
    """
    Translate image and bboxes on Y-axis.
    The pixel values filled in will be of the value replace.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p, pixels, replace=128):
        super().__init__()
        self.p = p
        self.pixels = pixels
        self.replace = replace

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            translate_image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, -self.pixels), fillcolor=(self.replace, self.replace, self.replace))
            if bboxs == None:
                return translate_image, bboxs
            else:
                translate_bbox = functional.translate_bbox(image, bboxs, self.pixels, self.replace, shift_horizontal=False)
                return translate_image, translate_bbox
        else:
            return image, bboxs
        
        
class Flip(torch.nn.Module):
    """
    Apply horizontal flip on image and bboxes.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            flip_image = ImageOps.mirror(image)
            if bboxs == None:
                return flip_image, bboxs
            else:
                flip_bbox = functional.flip(image, bboxs)
                return flip_image, flip_bbox
        else:
            image, bboxs
    
    
### Mask Augmentation
class Cutout(torch.nn.Module):
    """
    Apply cutout (https://arxiv.org/abs/1708.04552) to the image.
    This operation applies a (2*pad_size, 2*pad_size) mask of zeros to a random location within image.
    The pixel values filled in will be of the value replace.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, pad_size, replace=128):
        super().__init__()
        self.p = p
        self.pad_size = int(pad_size)
        self.replace = replace

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            cutout_image = functional.cutout(image, self.pad_size, self.replace)
            return cutout_image, bboxs
        else:
            return image, bboxs
    
    
### Color Augmentation based on BBoxes
class Equalize_Only_BBoxes(torch.nn.Module):
    """
    Apply equalize to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p/3

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            equalize_image = functional.equalize_only_bboxes(image, bboxs, self.p)
            return equalize_image, bboxs
        
        
class Solarize_Only_BBoxes(torch.nn.Module):
    """
    Apply solarize to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, threshold):
        super().__init__()
        self.p = p/3
        self.threshold = threshold

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            solarize_image = functional.solarize_only_bboxes(image, bboxs, self.p, self.threshold)
            return solarize_image, bboxs
    
    
### Geometric Augmentation based on BBoxes
class Rotate_Only_BBoxes(torch.nn.Module):
    """
    Apply rotation to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, degrees, replace=128):
        super().__init__()
        self.p = p/3
        self.degrees = degrees
        self.replace = replace

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            rotate_image = functional.rotate_only_bboxes(image, bboxs, self.p, self.degrees, self.replace)
            return rotate_image, bboxs
    
    
class ShearX_Only_BBoxes(torch.nn.Module):
    """
    Apply shear to each bboxes in the image with probability only on X-axis.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, level, replace=128):
        super().__init__()
        self.p = p/3
        self.level = level
        self.replace = replace

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            shear_image = functional.shear_only_bboxes(image, bboxs, self.p, self.level, self.replace, shift_horizontal=True)
            return shear_image, bboxs
    
    
class ShearY_Only_BBoxes(torch.nn.Module):
    """
    Apply shear to each bboxes in the image with probability only on Y-axis.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, level, replace=128):
        super().__init__()
        self.p = p/3
        self.level = level
        self.replace = replace

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            shear_image = functional.shear_only_bboxes(image, bboxs, self.p, self.level, self.replace, shift_horizontal=False)
            return shear_image, bboxs
    
    
class TranslateX_Only_BBoxes(torch.nn.Module):
    """
    Apply translation to each bboxes in the image with probability only on X-axis.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, pixels, replace=128):
        super().__init__()
        self.p = p/3
        self.pixels = pixels
        self.replace = replace

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            translate_image = functional.translate_only_bboxes(image, bboxs, self.p, self.pixels, self.replace, shift_horizontal=True)
            return translate_image, bboxs
    
    
class TranslateY_Only_BBoxes(torch.nn.Module):
    """
    Apply transloation to each bboxes in the image with probability only on Y-axis.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, pixels, replace=128):
        super().__init__()
        self.p = p/3
        self.pixels = pixels
        self.replace = replace

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            translate_image = functional.translate_only_bboxes(image, bboxs, self.p, self.pixels, self.replace, shift_horizontal=False)
            return translate_image, bboxs


class Flip_Only_BBoxes(torch.nn.Module):
    """
    Apply horizontal flip to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p/3

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            flip_image = functional.flip_only_bboxes(image, bboxs, self.p)
            return flip_image, bboxs

    
### Mask Augmentation based on BBoxes
class BBox_Cutout(torch.nn.Module):
    """
    Apply cutout to the image according to bbox information.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, pad_fraction, replace_with_mean=False):
        super().__init__()
        self.p = p
        self.pad_fraction = pad_fraction
        self.replace_with_mean = replace_with_mean

    def forward(self, image, bboxs):
        if (torch.rand(1) < self.p) and (bboxs != None):
            cutout_image = functional.bbox_cutout(image, bboxs, self.pad_fraction, self.replace_with_mean)
            return cutout_image, bboxs
        else:
            return image, bboxs


class Cutout_Only_BBoxes(torch.nn.Module):
    """
    Apply cutout to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, pad_size, replace=128):
        super().__init__()
        self.p = p/3
        self.pad_size = pad_size
        self.replace = replace

    def forward(self, image, bboxs):
        if bboxs == None:
            return image, bboxs
        else:
            cutout_image = functional.cutout_only_bboxes(image, bboxs, self.p, self.pad_size, self.replace)
            return cutout_image, bboxs