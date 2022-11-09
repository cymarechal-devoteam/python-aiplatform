# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Union

from mlflow import entities as mlflow_entities
from mlflow.store.tracking import abstract_store

from google.cloud import aiplatform
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import execution as execution_v1

# MLFlow RunStatus:
# https://www.mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunStatus
_MLFLOW_RUN_TO_VERTEX_RUN_STATUS = {
    mlflow_entities.RunStatus.FINISHED: execution_v1.Execution.State.COMPLETE,
    mlflow_entities.RunStatus.FAILED: execution_v1.Execution.State.FAILED,
    mlflow_entities.RunStatus.RUNNING: execution_v1.Execution.State.RUNNING,
    mlflow_entities.RunStatus.KILLED: execution_v1.Execution.State.CANCELLED,
    mlflow_entities.RunStatus.SCHEDULED: execution_v1.Execution.State.NEW,
}
mlflow_to_vertex_run_default = defaultdict(
    lambda: execution_v1.Execution.State.STATE_UNSPECIFIED
)
for mlflow_status in _MLFLOW_RUN_TO_VERTEX_RUN_STATUS:
    mlflow_to_vertex_run_default[mlflow_status] = _MLFLOW_RUN_TO_VERTEX_RUN_STATUS[
        mlflow_status
    ]

# Mapping of Vertex run status to MLFlow run status (inverse of _MLFLOW_RUN_TO_VERTEX_RUN_STATUS)
_VERTEX_RUN_TO_MLFLOW_RUN_STATUS = {
    v: k for k, v in _MLFLOW_RUN_TO_VERTEX_RUN_STATUS.items()
}
vertex_run_to_mflow_default = defaultdict(lambda: mlflow_entities.RunStatus.FAILED)
for vertex_status in _VERTEX_RUN_TO_MLFLOW_RUN_STATUS:
    vertex_run_to_mflow_default[vertex_status] = _VERTEX_RUN_TO_MLFLOW_RUN_STATUS[
        vertex_status
    ]


class _RunTracker(NamedTuple):
    """Tracks the current Vertex ExperimentRun.

    Stores the current ExperimentRun the plugin is writing to and whether or
    not this run is autocreated.

    Attributes:
        autocreate (bool):
            Whether the Vertex ExperimentRun should be autocreated. If False,
            the plugin writes to the currently active run created via
            `aiplatform.start_run()`.
        experiment_run (aiplatform.ExperimentRun):
            The currently set ExperimentRun.
    """

    autocreate: bool
    experiment_run: "aiplatform.ExperimentRun"


class _VertexMlflowTracking(abstract_store.AbstractStore):
    """Vertex plugin implementation of MLFlow's AbstractStore class."""

    def _to_mlflow_metric(
        self,
        vertex_metrics: Dict[str, Union[float, int, str]],
    ) -> Optional[List[mlflow_entities.Metric]]:
        """Helper method to convert Vertex metrics to mlflow.entities.Metric type.

        Args:
            vertex_metrics (Dict[str, Union[float, int, str]]):
                Required. A dictionary of Vertex metrics returned from
                ExperimentRun.get_metrics()
        Returns:
            List[mlflow_entities.Metric] - A list of metrics converted to MLFlow's
            Metric type.
        """

        mlflow_metrics = []

        if vertex_metrics:
            for metric_key in vertex_metrics:
                mlflow_metric = mlflow_entities.Metric(
                    key=metric_key,
                    value=vertex_metrics[metric_key],
                    step=0,
                    timestamp=0,
                )
                mlflow_metrics.append(mlflow_metric)
        else:
            return None

        return mlflow_metrics

    def _to_mlflow_params(
        self, vertex_params: Dict[str, Union[float, int, str]]
    ) -> Optional[mlflow_entities.Param]:
        """Helper method to convert Vertex params to mlflow.entities.Param type.

        Args:
            vertex_params (Dict[str, Union[float, int, str]]):
                Required. A dictionary of Vertex params returned from
                ExperimentRun.get_params()
        Returns:
            List[mlflow_entities.Param] - A list of params converted to MLFlow's
            Param type.
        """

        mlflow_params = []

        if vertex_params:
            for param_key in vertex_params:
                mlflow_param = mlflow_entities.Param(
                    key=param_key, value=vertex_params[param_key]
                )
                mlflow_params.append(mlflow_param)
        else:
            return None

        return mlflow_params

    def _to_mlflow_entity(
        self,
        vertex_exp: "aiplatform.Experiment",
        vertex_run: "aiplatform.ExperimentRun",
    ) -> mlflow_entities.Run:
        """Helper method to convert data to required MLFlow type.

        This converts data into MLFlow's mlflow_entities.Run type, which is a
        required return type for some methods we're overriding in this plugin.

        Args:
            vertex_exp (aiplatform.Experiment):
                Required. The current Vertex Experiment.
            vertex_run (aiplatform.ExperimentRun):
                Required. The active Vertex ExperimentRun
        Returns:
            mlflow_entities.Run - The data from the currently active run
            converted to MLFLow's mlflow_entities.Run type.

            https://www.mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run
        """

        run_info = mlflow_entities.RunInfo(
            run_id=vertex_run.name,
            run_uuid=vertex_run.name,
            experiment_id=vertex_exp.name,
            user_id="",
            status=vertex_run_to_mflow_default[vertex_run.state],
            start_time=1,
            end_time=2,
            lifecycle_stage=mlflow_entities.LifecycleStage.ACTIVE,
            artifact_uri="file:///tmp/",  # The plugin will fail if artifact_uri is not set to a valid filepath string
        )

        run_data = mlflow_entities.RunData(
            metrics=self._to_mlflow_metric(vertex_run.get_metrics()),
            params=self._to_mlflow_params(vertex_run.get_params()),
            tags={},
        )

        return mlflow_entities.Run(run_info=run_info, run_data=run_data)

    def __init__(self, store_uri: Optional[str], artifact_uri: Optional[str]) -> None:
        """Initializes the Vertex MLFlow plugin.

        This plugin overrides MLFlow's AbstractStore class to write metrics and
        parameters from model training code to Vertex Experiments. This plugin
        is private and should not be instantiated outside the Vertex SDK.

        The _run_map instance property is a dict mapping MLFlow run_id to an
        instance of _RunTracker with data on the corresponding Vertex
        ExperimentRun.

        For example: {
            'sklearn-12345': _RunTracker(autocreate=True, experiment_run=aiplatform.ExperimentRun(...))
        }

        Args:
            store_uri (str):
                The tracking store uri used by MLFlow to write parameters and
                metrics for a run. This plugin ignores store_uri since we are
                writing data to Vertex Experiments. For this plugin, the value
                of store_uri will always be `vertex-mlflow-plugin://`.
            artifact_uri (str):
                The artifact uri used by MLFlow to write artifacts generated by
                a run. This plugin ignores artifact_uri since it doesn't write
                any artifacts to Vertex.
        """

        self._run_map = {}
        self._vertex_experiment = None
        super(_VertexMlflowTracking, self).__init__()

    @property
    def run_map(self) -> Dict[str, Any]:
        return self._run_map

    @property
    def vertex_experiment(self) -> "aiplatform.Experiment":
        return self._vertex_experiment

    def create_run(
        self,
        experiment_id: str,
        user_id: str,
        start_time: str,
        tags: List[mlflow_entities.RunTag],
        run_name: str,
    ) -> mlflow_entities.Run:
        """Creates a new ExperimentRun in Vertex if no run is active.

        This overrides the behavior of MLFlow's `create_run()` method to check
        if there is a currently active ExperimentRun. If no ExperimentRun is
        active, a new Vertex ExperimentRun will be created with the name
        `<ml-framework>-<timestamp>`. If aiplatform.start_run() has been
        invoked and there is an active run, no run will be created and the
        currently active ExperimentRun will be returned as an MLFlow Run
        entity.

        Args:
            experiment_id (str):
                The ID of the currently set MLFlow Experiment. Not used by this
                plugin.
            user_id (str):
                The ID of the MLFlow user. Not used by this plugin.
            start_time (int):
                The start time of the run, in milliseconds since the UNIX
                epoch. Not used by this plugin.
            tags (List[mlflow_entities.RunTag]):
                The tags provided by MLFlow. Only the `mlflow.autologging` tag
                is used by this plugin.
            run_name (str):
                The name of the MLFlow run. Not used by this plugin.
        Returns:
            mlflow_entities.Run - The created run returned as MLFLow's run
            type.
        """

        self._vertex_experiment = (
            aiplatform.metadata.metadata._experiment_tracker._experiment
        )

        currently_active_run = (
            aiplatform.metadata.metadata._experiment_tracker._experiment_run
        )

        if currently_active_run:
            run_tracking_map = _RunTracker(
                autocreate=False, experiment_run=currently_active_run
            )
            current_run_id = currently_active_run.name

        # Create a new run if aiplatform.start_run() hasn't been called
        else:
            framework = ""

            for tag in tags:
                if tag.key == "mlflow.autologging":
                    framework = tag.value

            current_run_id = f"{framework}-{utils.timestamped_unique_name()}"
            autocreate_run = aiplatform.start_run(run=current_run_id)
            run_tracking_map = _RunTracker(
                autocreate=True, experiment_run=autocreate_run
            )

        self._run_map[current_run_id] = run_tracking_map

        return self._to_mlflow_entity(
            vertex_exp=self._vertex_experiment,
            vertex_run=run_tracking_map.experiment_run,
        )

    def update_run_info(
        self,
        run_id: str,
        run_status: mlflow_entities.RunStatus,
        end_time: int,
        run_name: str,
    ) -> mlflow_entities.RunInfo:
        """Updates the ExperimentRun status with the status provided by MLFlow.

        Args:
            run_id (str):
                The ID of the currently set MLFlow run. This is mapped to the
                corresponding ExperimentRun in self._run_map.
            run_status (mlflow_entities.RunStatus):
                The run status provided by MLFlow MLFlow.
            end_time (int):
                The end time of the run. Not used by this plugin.
            run_name (str):
                The name of the MLFlow run. Not used by this plugin.
        Returns:
            mlflow_entities.RunInfo - Info about the updated run in MLFlow's
            required RunInfo format.
        """

        if (
            run_status == mlflow_entities.RunStatus.FINISHED
            and not self._run_map[run_id].autocreate
        ):
            self._run_map[run_id].experiment_run.update_state(
                state=execution_v1.Execution.State.RUNNING
            )
        else:
            self._run_map[run_id].experiment_run.update_state(
                state=mlflow_to_vertex_run_default[run_status]
            )

        return mlflow_entities.RunInfo(
            run_uuid=run_id,
            run_id=run_id,
            status=run_status,
            end_time=end_time,
            experiment_id=self._vertex_experiment,
            user_id="",
            start_time=1,
            lifecycle_stage=mlflow_entities.LifecycleStage.ACTIVE,
            artifact_uri="file:///tmp/",
        )

    def log_batch(
        self,
        run_id: str,
        metrics: List[mlflow_entities.Metric],
        params: List[mlflow_entities.Param],
        tags: List[mlflow_entities.RunTag],
    ) -> None:
        """The primary logging method used by MLFlow.

        This plugin overrides this method to write the metrics and parameters
        provided by MLFlow to the active Vertex ExperimentRun.
        Args:
            run_id (str):
                The ID of the MLFlow run to write metrics to. This is mapped to
                the corresponding ExperimentRun in self._run_map.
            metrics (List[mlflow_entities.Metric]):
                A list of MLFlow metrics generated from the current model
                training run.
            params (List[mlflow_entities.Param]):
                A list of MLFlow params generated from the current model
                training run.
            tags (List[mlflow_entities.RunTag]):
                The tags provided by MLFlow. Not used by this plugin.
        """
        summary_metrics = {}
        summary_params = {}
        time_series_metrics = {}

        # Get the run to write to
        vertex_run = self._run_map[run_id].experiment_run

        for metric in metrics:
            if metric.step:
                if metric.step not in time_series_metrics:
                    time_series_metrics[metric.step] = {metric.key: metric.value}
                else:
                    time_series_metrics[metric.step][metric.key] = metric.value
            else:
                summary_metrics[metric.key] = metric.value

        for param in params:
            summary_params[param.key] = param.value

        if summary_metrics:
            vertex_run.log_metrics(metrics=summary_metrics)

        if summary_params:
            vertex_run.log_params(params=summary_params)

        # TODO(b/261722623): batch these calls
        if time_series_metrics:
            for step in time_series_metrics:
                vertex_run.log_time_series_metrics(time_series_metrics[step], step)

    def get_run(self, run_id: str) -> mlflow_entities.Run:
        """Gets the currently active run.

        Args:
            run_id (str):
                The ID of the currently set MLFlow run. This is mapped to the
                corresponding ExperimentRun in self._run_map.
        Returns:
            mlflow_entities.Run - The currently active Vertex ExperimentRun,
            returned as MLFLow's run type.
        """
        return self._to_mlflow_entity(
            vertex_exp=self._vertex_experiment,
            vertex_run=self._run_map[run_id].experiment_run,
        )
