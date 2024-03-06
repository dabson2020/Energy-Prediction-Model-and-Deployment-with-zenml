from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    model =  "XGB"
    print('Evaluating {} model'.format(model))
    model_name: str = model


       