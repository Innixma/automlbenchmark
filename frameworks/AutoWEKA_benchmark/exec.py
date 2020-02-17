import logging

import numpy as np

from autogluon_utils.benchmarking.baselines.autoweka.methods_autoweka import autoweka_fit_predict
from autogluon_utils.benchmarking.openml.automlbenchmark_wrapper import prepare_data

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** AutoWEKA Benchmark ****\n")

    X_train, y_train, X_test, y_test, problem_type, perf_metric = prepare_data(config=config, dataset=dataset)

    X_train['__label__'] = y_train
    num_models_trained, num_models_ensemble, fit_time, predictions, probabilities, predict_time, class_order = autoweka_fit_predict(
        train_data=X_train,
        test_data=X_test,
        label_column='__label__',
        problem_type=problem_type,
        output_directory='tmp_' + str(config.fold) + '/',
        eval_metric=perf_metric.name,
        runtime_sec=config.max_runtime_seconds,
        random_state=0,
        num_cores=config.cores,
    )

    print('baseline time:', fit_time)
    print('predict time:', predict_time)
    print('num_models_trained:', num_models_trained)
    print('num_models_ensemble:', num_models_ensemble)

    is_classification = config.type == 'classification'
    if is_classification & (len(probabilities.shape) == 1):
        probabilities = np.array([[1-row, row] for row in probabilities])

    classes = class_order

    if is_classification:
        print(classes)
        print(predictions[:5])
        print(probabilities[:5])
        print(y_test[:5])

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=False,
                             probabilities_labels=classes)

    return dict(
        models_count=num_models_trained,
        models_ensemble_count=num_models_ensemble,
        training_duration=fit_time,
        predict_duration=predict_time,
    )