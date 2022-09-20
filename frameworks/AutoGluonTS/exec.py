import logging
import os
import shutil
import warnings
import sys
import tempfile
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd

from autogluon.core.utils.savers import save_pd, save_pkl
from autogluon.tabular import TabularDataset
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon TimeSeries [v{__version__}] ****\n")

    timestamp_column = dataset.timestamp_column
    id_column = dataset.id_column
    prediction_length = dataset.prediction_length

    eval_metric = get_eval_metric(config)
    label = dataset.target.name
    time_limit = config.max_runtime_seconds

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    train_data, test_data = load_data(train_path=dataset.train.path,
                                      test_path=dataset.test.path,
                                      timestamp_column=timestamp_column,
                                      id_column=id_column)
    test_data_past = test_data.copy().slice_by_timestep(slice(None, -prediction_length))

    predictor_path = tempfile.mkdtemp() + os.sep
    with Timer() as training:
        predictor = TimeSeriesPredictor(
            target=label,
            path=predictor_path,
            prediction_length=prediction_length,
            eval_metric=eval_metric,
        )
        predictor.fit(
            train_data=train_data,
            time_limit=time_limit,
            **training_params,
        )

    with Timer() as predict:
        predictions = predictor.predict(test_data_past)
    log.info(predictions)

    predictions_only = predictions['mean'].values
    test_data_future = test_data.copy().slice_by_timestep(slice(-prediction_length, None))
    truth_only = test_data_future[label].values

    log.info(predictions_only)
    log.info(truth_only)

    leaderboard = predictor.leaderboard(test_data, silent=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)

    save_artifacts(predictor=predictor, leaderboard=leaderboard, config=config)
    shutil.rmtree(predictor.path, ignore_errors=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions_only,
                  truth=truth_only,
                  probabilities=None,
                  probabilities_labels=None,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  quantiles=predictions.drop(columns=['mean']))


def load_data(train_path, test_path, timestamp_column, id_column):

    train_df = pd.read_csv(
        train_path,
        parse_dates=[timestamp_column],
    )

    train_data = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    test_df = pd.read_csv(
        test_path,
        parse_dates=[timestamp_column],
    )

    test_data = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    return train_data, test_data


def get_eval_metric(config):
    # TODO: Support more metrics
    metrics_mapping = dict(
        mse="MSE",
        rmse="RMSE",
        mase="MASE",
    )

    eval_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if eval_metric is None:
        log.warning("Performance metric %s not supported.", config.metric)
    return eval_metric


def save_artifacts(predictor, leaderboard, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            leaderboard_dir = output_subdir("leaderboard", config)
            save_pd.save(path=os.path.join(leaderboard_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            models_dir = output_subdir("models", config)
            zip_path(predictor.path, os.path.join(models_dir, "models.zip"))
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)