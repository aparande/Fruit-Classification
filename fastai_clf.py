from fastai.metrics import error_rate
from fastai.vision import *
import matplotlib.pyplot as plt

data_bunch = ImageDataBunch.from_folder("fruits-360", train="Training", test="Test", ds_tfms=get_transforms(), size=24)

model = create_cnn(data_bunch, models.resnet34, metrics=error_rate)
model.fit_one_cycle(1)
learn.save('stage-1')