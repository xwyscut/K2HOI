
from .models_k2hoi.hoiclip import build as build_models_swifthoiclip
from .visualmodel.hoiclip import build as visualization

def build_model(args):
    if args.model_name == "SWIFTHOICLIP": 
        return build_models_swifthoiclip(args)
    else :
        return visualization(args)

