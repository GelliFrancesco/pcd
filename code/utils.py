import torch
from model import VggModel
import pandas as pd


def load_model(path, num_brands, cuda=False):
    the_model = VggModel(num_brands)
    if cuda:
        the_model.load_state_dict(torch.load(path))
    else:
        the_model.load_state_dict(torch.load(path, map_location=lambda gpu, loc: gpu))
    return the_model


def build_brand_list(path):
    brand_list = pd.read_csv(path)
    return brand_list[['username','vertical']]


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
