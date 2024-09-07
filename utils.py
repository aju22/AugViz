import inspect
from typing import Callable, Union, get_args, get_origin
import albumentations as A

FILTER_TRANSFORMS = [
    A.ImageOnlyTransform,
    A.DualTransform,
    A.ReferenceBasedTransform,
    A.TemplateTransform,
    A.Lambda,
]

def is_not_supported_transform(transform_cls):
    sig = inspect.signature(transform_cls)
    
    if issubclass(transform_cls, A.ReferenceBasedTransform):
        return True
    
    for filter_transform_cls in FILTER_TRANSFORMS:
        if transform_cls is filter_transform_cls:
            return True
    
    for param in sig.parameters.values():
        if issubclass(type(param.annotation), type(Callable)):
            return True
        if param.name in ["read_fn", "reference_images"]:
            return True
    
    return False

def get_transforms_map():
    transforms_map = {
        name: cls
        for name, cls in vars(A).items()
        if (
            inspect.isclass(cls)
            and issubclass(cls, (A.DualTransform, A.ImageOnlyTransform))
            and not is_not_supported_transform(cls)
        )
    }
    transforms_map.pop("DualTransform", None)
    transforms_map.pop("ImageOnlyTransform", None)
    transforms_map.pop("ReferenceBasedTransform", None)
    transforms_map.pop("ToFloat", None)
    transforms_map.pop("Normalize", None)
    return transforms_map

def apply_augmentations(image, augmentations):
    aug = A.Compose(augmentations)
    try:
        augmented = aug(image=image)
        return augmented['image']
    except Exception as e:
        return None, str(e)

def is_range_param(param_value):
    return isinstance(param_value, tuple) and len(param_value) == 2 and (param_value[0] != param_value[1]) and all(isinstance(x, (int, float)) for x in param_value)



def get_params_info(aug_class):
    signature = inspect.signature(aug_class)
    params_info = {}
    for name, param in signature.parameters.items():
        params_info[name] = {
            'default': param.default if param.default is not inspect.Parameter.empty else None,
            'annotation': param.annotation if param.annotation is not inspect.Parameter.empty else None,
            'required': param.default is inspect.Parameter.empty
        }
    return params_info

def infer_type_from_annotation(annotation):
    if annotation is None:
        return str
    if get_origin(annotation) is Union:
        types = get_args(annotation)
        if type(None) in types:
            types = [t for t in types if t is not type(None)]
        return types[0] if types else str
    return annotation