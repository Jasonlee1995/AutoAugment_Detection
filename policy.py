import torch, random
from augmentation import *


M = 10

color_range = torch.arange(0, 0.9+1e-8, (0.9-0)/M).tolist()
rotate_range = torch.arange(0, 30+1e-8, (30-0)/M).tolist()
shear_range = torch.arange(0, 0.3+1e-8, (0.3-0)/M).tolist()
translate_range = torch.arange(0, 250+1e-8, (250-0)/M).tolist()
translate_bbox_range = torch.arange(0, 120+1e-8, (120-0)/M).tolist()


Mag = {'Brightness' : color_range, 'Color' : color_range, 'Contrast' : color_range, 
       'Posterize' : torch.arange(4, 8+1e-8, (8-4)/M).tolist()[::-1], 'Sharpness' : color_range, 
       'Solarize' : torch.arange(0, 256+1e-8, (256-0)/M).tolist()[::-1], 'SolarizeAdd' : torch.arange(0, 110+1e-8, (110-0)/M).tolist(),
       
       'Cutout' : torch.arange(0, 100+1e-8, (100-0)/M).tolist(),
       
       'Rotate_BBox' : rotate_range, 'ShearX_BBox' : shear_range, 'ShearY_BBox' : shear_range,
       'TranslateX_BBox' : translate_range, 'TranslateY_BBox' : translate_range,
           
       'Rotate_Only_BBoxes' : rotate_range, 'ShearX_Only_BBoxes' : shear_range, 'ShearY_Only_BBoxes' : shear_range,
       'TranslateX_Only_BBoxes' : translate_bbox_range, 'TranslateY_Only_BBoxes' : translate_bbox_range,
       
       'Solarize_Only_BBoxes' : torch.arange(0, 256+1e-8, (256-0)/M).tolist()[::-1],
       
       'BBox_Cutout' : torch.arange(0, 0.75+1e-8, (0.75-0)/M).tolist(), 'Cutout_Only_BBoxes' : torch.arange(0, 50+1e-8, (50-0)/M).tolist()
      }


Fun = {'AutoContrast' : AutoContrast, 'Brightness' : Brightness, 'Color' : Color, 'Contrast' : Contrast, 'Equalize' : Equalize, 
       'Posterize' : Posterize, 'Sharpness' : Sharpness, 'Solarize' : Solarize, 'SolarizeAdd' : SolarizeAdd,
       
       'Cutout' : Cutout,
       
       'Rotate_BBox' : Rotate_BBox, 'ShearX_BBox' : ShearX_BBox, 'ShearY_BBox' : ShearY_BBox,
       'TranslateX_BBox' : TranslateX_BBox, 'TranslateY_BBox' : TranslateY_BBox,
           
       'Rotate_Only_BBoxes' : Rotate_Only_BBoxes, 'ShearX_Only_BBoxes' : ShearX_Only_BBoxes, 'ShearY_Only_BBoxes' : ShearY_Only_BBoxes,
       'TranslateX_Only_BBoxes' : TranslateX_Only_BBoxes, 'TranslateY_Only_BBoxes' : TranslateY_Only_BBoxes, 'Flip_Only_BBoxes' : Flip_Only_BBoxes,
       
       'Equalize_Only_BBoxes' : Equalize_Only_BBoxes, 'Solarize_Only_BBoxes' : Solarize_Only_BBoxes,
       
       'BBox_Cutout' : BBox_Cutout, 'Cutout_Only_BBoxes' : Cutout_Only_BBoxes
      }


class Policy(torch.nn.Module):
    def __init__(self, policy, pre_transform, post_transform):
        super().__init__()
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        
        if policy == 'policy_v0': self.policy = policy_v0()
        elif policy == 'policy_v1': self.policy = policy_v1()
        elif policy == 'policy_v2': self.policy = policy_v2()
        elif policy == 'policy_v3': self.policy = policy_v3()
        elif policy == 'policy_vtest': self.policy = policy_vtest()

    def forward(self, image, bboxs):
        policy_idx = random.randint(0, len(self.policy)-1)
        policy_transform = self.pre_transform + self.policy[policy_idx] + self.post_transform
        policy_transform = Compose(policy_transform)
        image, bboxs = policy_transform(image, bboxs)
        return image, bboxs
    
    
def SubPolicy(f1, p1, m1, f2, p2, m2):
    subpolicy = []
    if f1 in ['AutoContrast', 'Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes']: subpolicy.append(Fun[f1](p1))
    else: subpolicy.append(Fun[f1](p1, Mag[f1][m1]))
    
    if f2 in ['AutoContrast', 'Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes']: subpolicy.append(Fun[f2](p2))
    else: subpolicy.append(Fun[f2](p2, Mag[f2][m2]))
        
    return subpolicy


def SubPolicy3(f1, p1, m1, f2, p2, m2, f3, p3, m3):
    subpolicy = []
    if f1 in ['AutoContrast', 'Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes']: subpolicy.append(Fun[f1](p1))
    else: subpolicy.append(Fun[f1](p1, Mag[f1][m1]))
    
    if f2 in ['AutoContrast', 'Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes']: subpolicy.append(Fun[f2](p2))
    else: subpolicy.append(Fun[f2](p2, Mag[f2][m2]))
        
    if f3 in ['AutoContrast', 'Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes']: subpolicy.append(Fun[f3](p3))
    else: subpolicy.append(Fun[f3](p3, Mag[f3][m3]))
        
    return subpolicy  
    

def policy_v0():
    policy = [SubPolicy('TranslateX_BBox', 0.6, 4,           'Equalize', 0.8, None),
              SubPolicy('TranslateY_Only_BBoxes', 0.2, 2,    'Cutout', 0.8, 8),
              SubPolicy('Sharpness', 0.0, 8,                 'ShearX_BBox', 0.4, 0),
              SubPolicy('ShearY_BBox', 1.0, 2,               'TranslateY_Only_BBoxes', 0.6, 6),
              SubPolicy('Rotate_BBox', 0.6, 10,              'Color', 1.0, 6)]
    return policy


def policy_v1():
    policy = [SubPolicy('TranslateX_BBox', 0.6, 4,           'Equalize', 0.8, None),
              SubPolicy('TranslateY_Only_BBoxes', 0.2, 2,    'Cutout', 0.8, 8),
              SubPolicy('Sharpness', 0, 8,                   'ShearX_BBox', 0.4, 0),
              SubPolicy('ShearY_BBox', 1.0, 2,               'TranslateY_Only_BBoxes', 0.6, 6),
              SubPolicy('Rotate_BBox', 0.6, 10,              'Color', 1.0, 6),
              SubPolicy('Color', 0.0, 0,                     'ShearX_Only_BBoxes', 0.8, 4),
              SubPolicy('ShearY_Only_BBoxes', 0.8, 2,        'Flip_Only_BBoxes', 0.0, None),
              SubPolicy('Equalize', 0.6, None,               'TranslateX_BBox', 0.2, 2),
              SubPolicy('Color', 1.0, 10,                    'TranslateY_Only_BBoxes', 0.4, 6),
              SubPolicy('Rotate_BBox', 0.8, 10,              'Contrast', 0.0, 10),
              SubPolicy('Cutout', 0.2, 2,                    'Brightness', 0.8, 10),
              SubPolicy('Color', 1.0, 6,                     'Equalize', 1.0, None),
              SubPolicy('Cutout_Only_BBoxes', 0.4, 6,        'TranslateY_Only_BBoxes', 0.8, 2),
              SubPolicy('Color', 0.2, 8,                     'Rotate_BBox', 0.8, 10),
              SubPolicy('Sharpness', 0.4, 4,                 'TranslateY_Only_BBoxes', 0.0, 4),
              SubPolicy('Sharpness', 1.0, 4,                 'SolarizeAdd', 0.4, 4),
              SubPolicy('Rotate_BBox', 1.0, 8,               'Sharpness', 0.2, 8),
              SubPolicy('ShearY_BBox', 0.6, 10,              'Equalize_Only_BBoxes', 0.6, None),
              SubPolicy('ShearX_BBox', 0.2, 6,               'TranslateY_Only_BBoxes', 0.2, 10),
              SubPolicy('SolarizeAdd', 0.6, 8,               'Brightness', 0.8, 10)]
    return policy


def policy_vtest():
    policy = [SubPolicy('TranslateX_BBox', 1.0, 4,           'Equalize', 1.0, None)]
    return policy


def policy_v2():
    policy = [SubPolicy3('Color', 0.0, 6,                    'Cutout', 0.6, 8,                 'Sharpness', 0.4, 8),
              SubPolicy3('Rotate_BBox', 0.4, 8,              'Sharpness', 0.4, 2,              'Rotate_BBox', 0.8, 10),
              SubPolicy('TranslateY_BBox', 1.0, 8,           'AutoContrast', 0.8, None),
              SubPolicy3('AutoContrast', 0.4, None,          'ShearX_BBox', 0.8, 8,            'Brightness', 0.0, 10),
              SubPolicy3('SolarizeAdd', 0.2, 6,              'Contrast', 0.0, 10,              'AutoContrast', 0.6, None),
              SubPolicy3('Cutout', 0.2, 0,                   'Solarize', 0.8, 8,               'Color', 1.0, 4),
              SubPolicy3('TranslateY_BBox', 0.0, 4,          'Equalize', 0.6, None,            'Solarize', 0.0, 10),
              SubPolicy3('TranslateY_BBox', 0.2, 2,          'ShearY_BBox', 0.8, 8,            'Rotate_BBox', 0.8, 8),
              SubPolicy3('Cutout', 0.8, 8,                   'Brightness', 0.8, 8,             'Cutout', 0.2, 2),
              SubPolicy3('Color', 0.8, 4,                    'TranslateY_BBox', 1.0, 6,        'Rotate_BBox', 0.6, 6),
              SubPolicy3('Rotate_BBox', 0.6, 10,             'BBox_Cutout', 1.0, 4,            'Cutout', 0.2, 8),
              SubPolicy3('Rotate_BBox', 0.0, 0,              'Equalize', 0.6, None,            'ShearY_BBox', 0.6, 8),
              SubPolicy3('Brightness', 0.8, 8,               'AutoContrast', 0.4, None,        'Brightness', 0.2, 2),
              SubPolicy3('TranslateY_BBox', 0.4, 8,          'Solarize', 0.4, 6,               'SolarizeAdd', 0.2, 10),
              SubPolicy3('Contrast', 1.0, 10,                'SolarizeAdd', 0.2, 8,            'Equalize', 0.2, None)]
    return policy

              
def policy_v3():
    policy = [SubPolicy('Posterize', 0.8, 2,                 'TranslateX_BBox', 1.0, 8),
              SubPolicy('BBox_Cutout', 0.2, 10,              'Sharpness', 1.0, 8),
              SubPolicy('Rotate_BBox', 0.6, 8,               'Rotate_BBox', 0.8, 10),
              SubPolicy('Equalize', 0.8, None,               'AutoContrast', 0.2, None),
              SubPolicy('SolarizeAdd', 0.2, 2,               'TranslateY_BBox', 0.2, 8),
              SubPolicy('Sharpness', 0.0, 2,                 'Color', 0.4, 8),
              SubPolicy('Equalize', 1.0, None,               'TranslateY_BBox', 1.0, 8),
              SubPolicy('Posterize', 0.6, 2,                 'Rotate_BBox', 0.0, 10),
              SubPolicy('AutoContrast', 0.6, None,           'Rotate_BBox', 1.0, 6),
              SubPolicy('Equalize', 0.0, None,               'Cutout', 0.8, 10),
              SubPolicy('Brightness', 1.0, 2,                'TranslateY_BBox', 1.0, 6),
              SubPolicy('Contrast', 0.0, 2,                  'ShearY_BBox', 0.8, 0),
              SubPolicy('AutoContrast', 0.8, None,           'Contrast', 0.2, 10),
              SubPolicy('Rotate_BBox', 1.0, 10,              'Cutout', 1.0, 10),
              SubPolicy('SolarizeAdd', 0.8, 6,               'Equalize', 0.8, None)]
    return policy