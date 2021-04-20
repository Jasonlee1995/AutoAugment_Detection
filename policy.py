import torch
from augmentation import *


M = 10

color_range = torch.arange(0.1, 1.9+1e-10, (1.9-0.1)/M).tolist()

def rotate_range():
    if torch.rand(1) < 0.5: return torch.arange(0, 30+1e-10, (30-0)/M).tolist()
    return torch.arange(0, -30-1e-10, (-30-0)/M).tolist()

def shear_range():
    if torch.rand(1) < 0.5: return torch.arange(0, 0.3+1e-10, (0.3-0)/M).tolist()
    return torch.arange(0, -0.3-1e-10, (-0.3-0)/M).tolist()

def translate_range():
    if torch.rand(1) < 0.5: return torch.arange(0, 250+1e-10, (250-0)/M).tolist()
    return torch.arange(0, -250-1e-10, (-250-0)/M).tolist()

def translate_bbox_range():
    if torch.rand(1) < 0.5: return torch.arange(0, 120+1e-10, (120-0)/M).tolist()
    return torch.arange(0, -120-1e-10, (-120-0)/M).tolist()

Mag = {'Brightness' : color_range, 'Color' : color_range, 'Contrast' : color_range, 
       'Posterize' : torch.arange(4, 8+1e-10, (8-4)/M).tolist(), 'Sharpness' : color_range, 
       'Solarize' : torch.arange(0, 256+1e-10, (256-0)/M).tolist(), 'SolarizeAdd' : torch.arange(0, 110+1e-10, (110-0)/M).tolist(),
       
       'Cutout' : torch.arange(0, 100+1e-10, (100-0)/M).tolist(),
       
       'Rotate_BBox' : rotate_range, 'ShearX_BBox' : shear_range, 'ShearY_BBox' : shear_range,
       'TranslateX_BBox' : translate_range, 'TranslateY_BBox' : translate_range,
           
       'Rotate_Only_BBoxes' : rotate_range, 'ShearX_Only_BBoxes' : shear_range, 'ShearY_Only_BBoxes' : shear_range,
       'TranslateX_Only_BBoxes' : translate_bbox_range, 'TranslateY_Only_BBoxes' : translate_bbox_range,
       
       'Solarize_Only_BBoxes' : torch.arange(0, 256+1e-10, (256-0)/M).tolist(),
       
       'BBox_Cutout' : torch.arange(0, 0.75+1e-10, (0.75-0)/M).tolist(), 'Cutout_Only_BBoxes' : torch.arange(0, 50+1e-10, (50-0)/M).tolist()
      }


def policy_v0():
    policy = [TranslateX_BBox(0.6, Mag['TranslateX_BBox']()[4]), Equalize(0.8), 
              TranslateY_Only_BBoxes(0.2, Mag['TranslateY_Only_BBoxes']()[2]), Cutout(0.8, Mag['Cutout'][8]),
              Sharpness(0.0, Mag['Sharpness'][8]), ShearX_BBox(0.4, Mag['ShearX_BBox']()[0]),
              ShearY_BBox(1.0, Mag['ShearY_BBox']()[2]), TranslateY_Only_BBoxes(0.6, Mag['TranslateY_Only_BBoxes']()[6]),
              Rotate_BBox(0.6, Mag['Rotate_BBox']()[10]), Color(1.0, Mag['Color'][6])]
    return policy


def policy_v1():
    policy = [TranslateX_BBox(0.6, Mag['TranslateX_BBox']()[4]), Equalize(0.8),
              TranslateY_Only_BBoxes(0.2, Mag['TranslateY_Only_BBoxes']()[2]), Cutout(0.8, Mag['Cutout'][8]),
              Sharpness(0.0, Mag['Sharpness'][8]), ShearX_BBox(0.4, Mag['ShearX_BBox']()[0]),
              ShearY_BBox(1.0, Mag['ShearY_BBox']()[2]), TranslateY_Only_BBoxes(0.6, Mag['TranslateY_Only_BBoxes']()[6]),
              Rotate_BBox(0.6, Mag['Rotate_BBox']()[10]), Color(1.0, Mag['Color'][6]),
              Color(0.0, Mag['Color'][0]), ShearX_Only_BBoxes(0.8, Mag['ShearX_Only_BBoxes']()[4]),
              ShearY_Only_BBoxes(0.8, Mag['ShearY_Only_BBoxes']()[2]), Flip_Only_BBoxes(0.0),
              Equalize(0.6), TranslateX_BBox(0.2, Mag['TranslateX_BBox']()[2]),
              Color(1.0, Mag['Color'][10]), TranslateY_Only_BBoxes(0.4, Mag['TranslateY_Only_BBoxes']()[6]),
              Rotate_BBox(0.8, Mag['Rotate_BBox']()[10]), Contrast(0.0, Mag['Contrast'][10]),
              Cutout(0.2, Mag['Cutout'][2]), Brightness(0.8, Mag['Brightness'][10]),
              Color(1.0, Mag['Color'][6]), Equalize(1.0),
              Cutout_Only_BBoxes(0.4, Mag['Cutout_Only_BBoxes'][6]), TranslateY_Only_BBoxes(0.8, Mag['TranslateY_Only_BBoxes']()[2]),
              Color(0.2, Mag['Color'][8]), Rotate_BBox(0.8, Mag['Rotate_BBox']()[10]),
              Sharpness(0.4, Mag['Sharpness'][4]), TranslateY_Only_BBoxes(0.0, Mag['TranslateY_Only_BBoxes']()[4]),
              Sharpness(1.0, Mag['Sharpness'][4]), SolarizeAdd(0.4, Mag['SolarizeAdd'][4]),
              Rotate_BBox(1.0, Mag['Rotate_BBox']()[8]), Sharpness(0.2, Mag['Sharpness'][8]),
              ShearY_BBox(0.6, Mag['ShearY_BBox']()[10]), Equalize_Only_BBoxes(0.6),
              ShearX_BBox(0.2, Mag['ShearX_BBox']()[6]), TranslateY_Only_BBoxes(0.2, Mag['TranslateY_Only_BBoxes']()[10]),
              SolarizeAdd(0.6, Mag['SolarizeAdd'][8]), Brightness(0.8, Mag['Brightness'][10])]
    return policy


def policy_vtest():
    policy = [TranslateX_BBox(1.0, Mag['TranslateX_BBox']()[4]), Equalize(1.0)]
    return policy


def policy_v2():
    policy = [Color(0.0, Mag['Color'][6]), Cutout(0.6, Mag['Cutout'][8]), Sharpness(0.4, Mag['Sharpness'][8]),
              Rotate_BBox(0.4, Mag['Rotate_BBox']()[8]), Sharpness(0.4, Mag['Sharpness'][2]), Rotate_BBox(0.8, Mag['Rotate_BBox']()[10]), 
              TranslateY_BBox(1.0, Mag['TranslateY_BBox']()[8]), AutoContrast(0.8),
              AutoContrast(0.4), ShearX_BBox(0.8, Mag['ShearX_BBox']()[8]), Brightness(0.0, Mag['Brightness'][10]),
              SolarizeAdd(0.2, Mag['SolarizeAdd'][6]), Contrast(0.0, Mag['Contrast'][10]), AutoContrast(0.6), 
              Cutout(0.2, Mag['Cutout'][0]), Solarize(0.8, Mag['Solarize'][8]), Color(1.0, Mag['Color'][4]), 
              TranslateY_BBox(0.0, Mag['TranslateY_BBox']()[4]), Equalize(0.6), Solarize(0.0, Mag['Solarize'][10]), 
              TranslateY_BBox(0.2, Mag['TranslateY_BBox']()[2]), ShearY_BBox(0.8, Mag['ShearY_BBox']()[8]), Rotate_BBox(0.8, Mag['Rotate_BBox']()[8]), 
              Cutout(0.8, Mag['Cutout'][8]), Brightness(0.8, Mag['Brightness'][8]), Cutout(0.2, Mag['Cutout'][2]),
              Color(0.8, Mag['Color'][4]), TranslateY_BBox(1.0, Mag['TranslateY_BBox']()[6]), Rotate_BBox(0.6, Mag['Rotate_BBox']()[6]), 
              Rotate_BBox(0.6, Mag['Rotate_BBox']()[10]), BBox_Cutout(1.0, Mag['BBox_Cutout'][4]), Cutout(0.2, Mag['Cutout'][8]), 
              Rotate_BBox(0.0, Mag['Rotate_BBox']()[0]), Equalize(0.6), ShearY_BBox(0.6, Mag['ShearY_BBox']()[8]), 
              Brightness(0.8, Mag['Brightness'][8]), AutoContrast(0.4), Brightness(0.2, Mag['Brightness'][2]), 
              TranslateY_BBox(0.4, Mag['TranslateY_BBox']()[8]), Solarize(0.4, Mag['Solarize'][6]), SolarizeAdd(0.2, Mag['SolarizeAdd'][10]), 
              Contrast(1.0, Mag['Contrast'][10]), SolarizeAdd(0.2, Mag['SolarizeAdd'][8]), Equalize(0.2)]
    return policy


def policy_v3():
    policy = [Posterize(0.8, Mag['Posterize'][2]), TranslateX_BBox(1.0, Mag['TranslateX_BBox']()[8]), 
              BBox_Cutout(0.2, Mag['BBox_Cutout'][10]), Sharpness(1.0, Mag['Sharpness'][8]), 
              Rotate_BBox(0.6, Mag['Rotate_BBox']()[8]), Rotate_BBox(0.8, Mag['Rotate_BBox']()[10]), 
              Equalize(0.8), AutoContrast(0.2),
              SolarizeAdd(0.2, Mag['SolarizeAdd'][2]), TranslateY_BBox(0.2, Mag['TranslateY_BBox']()[8]), 
              Sharpness(0.0, Mag['Sharpness'][2]), Color(0.4, Mag['Color'][8]), 
              Equalize(1.0), TranslateY_BBox(1.0, Mag['TranslateY_BBox']()[8]), 
              Posterize(0.6, Mag['Posterize'][2]), Rotate_BBox(0.0, Mag['Rotate_BBox']()[10]), 
              AutoContrast(0.6), Rotate_BBox(1.0, Mag['Rotate_BBox']()[6]), 
              Equalize(0.0), Cutout(0.8, Mag['Cutout'][10]), 
              Brightness(1.0, Mag['Brightness'][2]), TranslateY_BBox(1.0, Mag['TranslateY_BBox']()[6]), 
              Contrast(0.0, Mag['Contrast'][2]), ShearY_BBox(0.8, Mag['ShearY_BBox']()[0]), 
              AutoContrast(0.8), Contrast(0.2, Mag['Contrast'][10]), 
              Rotate_BBox(1.0, Mag['Rotate_BBox']()[10]), Cutout(1.0, Mag['Cutout'][10]), 
              SolarizeAdd(0.8, Mag['SolarizeAdd'][6]), Equalize(0.8)]
    return policy