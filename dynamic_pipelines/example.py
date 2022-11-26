from typing import Dict, List, Any

import pandas as pd
from pandas import DataFrame
from zenml.steps import Output, step, BaseStep, BaseParameters

from dynamic_pipelines.dynamic_pipeline import DynamicPipeline


class StepsConfiguration(BaseParameters):
    product: str
    model_id: int = 0


@step
def calc_features(conf: StepsConfiguration) -> Output(my_output=DataFrame):
    return pd.DataFrame({f"col_of_product_{conf.product}": [0]})


@step
def preprocess_data(conf: StepsConfiguration, data: DataFrame) -> Output(my_output=DataFrame):
    return data + conf.model_id


@step
def predict_data(data: DataFrame) -> Output(my_output=DataFrame):
    data['prediction'] = 17 * data.iloc[:, 0] + 31
    return data


@step
def store_data(data: DataFrame) -> None:
    print("pretend to store data:")
    print(data)


class MyPipeline(DynamicPipeline):
    def __init__(self, model_count, product_name: str, **kwargs: Any):
        self.model_count = model_count
        self.product_name = product_name
        super().__init__(**kwargs)

    def initialize_steps(self) -> List[BaseStep]:
        calc_feature_step = calc_features(StepsConfiguration(product=self.product_name))
        preprocess_steps = [self.create_step(preprocess_data, i)(StepsConfiguration(
            product=self.product_name,
            model_id=i)) for i in range(self.model_count)]
        prediction_steps = [self.create_step(predict_data, i)() for i in range(self.model_count)]
        store_steps = [self.create_step(store_data, i)() for i in range(self.model_count)]
        return [calc_feature_step] + preprocess_steps + prediction_steps + store_steps

    def connect(self, **steps) -> None:
        calc_features_step = self.get_step(calc_features)
        data = calc_features_step()
        for i in range(self.model_count):
            preprocess_step = self.get_step(preprocess_data, i)
            preprocessed_data = preprocess_step(data)

            predict_step = self.get_step(predict_data, i)
            predicted_data = predict_step(preprocessed_data)

            store_step = self.get_step(store_data, i)
            store_step(predicted_data)


def define_my_pipelines(conf: Dict[str, int]):
    for product_name, number_of_models in conf.items():
        product_pipeline = MyPipeline.as_template_of(f"{product_name}_pipeline")
        yield product_pipeline(model_count=number_of_models, product_name=product_name)



if __name__ == '__main__':
    for pipeline in define_my_pipelines(conf={'sm': 5, 'cc': 4}):
        pipeline.run()
    # # MyPipeline(model_count=5, product_name='sm').run()
    # RFHyperParameterTuning(RandomForestClassifierParameters(n_estimators=100),
    #                        RandomForestClassifierParameters(n_estimators=200)).run()
