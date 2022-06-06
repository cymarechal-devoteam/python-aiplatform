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

from typing import Optional, Dict, List, Sequence

import proto

from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.compat.types import context as gca_context
from google.cloud.aiplatform.compat.types import (
    lineage_subgraph as gca_lineage_subgraph,
)
from google.cloud.aiplatform.compat.types import (
    metadata_service as gca_metadata_service,
)
from google.cloud.aiplatform.metadata import artifact
from google.cloud.aiplatform.metadata import execution
from google.cloud.aiplatform.metadata import resource


class _Context(resource._Resource):
    """Metadata Context resource for Vertex AI"""

    _resource_noun = "contexts"
    _getter_method = "get_context"
    _delete_method = "delete_context"
    _parse_resource_name_method = "parse_context_path"
    _format_resource_name_method = "context_path"
    _list_method = "list_contexts"

    @property
    def parent_contexts(self) -> Sequence[str]:
        """The parent context resource names of this context."""
        return self.gca_resource.parent_contexts

    def add_artifacts_and_executions(
        self,
        artifact_resource_names: Optional[Sequence[str]] = None,
        execution_resource_names: Optional[Sequence[str]] = None,
    ):
        """Associate Executions and attribute Artifacts to a given Context.

        Args:
            artifact_resource_names (Sequence[str]):
                Optional. The full resource name of Artifacts to attribute to the Context.
            execution_resource_names (Sequence[str]):
                Optional. The full resource name of Executions to associate with the Context.
        """
        self.api_client.add_context_artifacts_and_executions(
            context=self.resource_name,
            artifacts=artifact_resource_names,
            executions=execution_resource_names,
        )

    def get_artifacts(self) -> List[artifact.Artifact]:
        """Returns all Artifact associated to this Context.

        Returns:
            artifacts(List[Artifacts]): All Artifacts under this context.
        """
        return artifact.Artifact.list(
            filter=metadata_utils._make_filter_string(in_context=[self.resource_name]),
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )

    @classmethod
    def _create_resource(
        cls,
        client: utils.MetadataClientWithOverride,
        parent: str,
        resource_id: str,
        schema_title: str,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> proto.Message:
        gapic_context = gca_context.Context(
            schema_title=schema_title,
            schema_version=schema_version,
            display_name=display_name,
            description=description,
            metadata=metadata if metadata else {},
        )
        return client.create_context(
            parent=parent,
            context=gapic_context,
            context_id=resource_id,
        )

    @classmethod
    def _update_resource(
        cls,
        client: utils.MetadataClientWithOverride,
        resource: proto.Message,
    ) -> proto.Message:
        """Update Contexts with given input.

        Args:
            client (utils.MetadataClientWithOverride):
                Required. client to send require to Metadata Service.
            resource (proto.Message):
                Required. The proto.Message which contains the update information for the resource.
        """

        return client.update_context(context=resource)

    @classmethod
    def _list_resources(
        cls,
        client: utils.MetadataClientWithOverride,
        parent: str,
        filter: Optional[str] = None,
    ):
        """List Contexts in the parent path that matches the filter.

        Args:
            client (utils.MetadataClientWithOverride):
                Required. client to send require to Metadata Service.
            parent (str):
                Required. The path where Contexts are stored.
            filter (str):
                Optional. filter string to restrict the list result
        """

        list_request = gca_metadata_service.ListContextsRequest(
            parent=parent,
            filter=filter,
        )
        return client.list_contexts(request=list_request)

    def add_context_children(self, contexts: List["_Context"]):
        """Adds the provided contexts as children of this context.

        Args:
            contexts (List[_Context]): Contexts to add as children.
        """
        self.api_client.add_context_children(
            context=self.resource_name,
            child_contexts=[c.resource_name for c in contexts],
        )

    def query_lineage_subgraph(self) -> gca_lineage_subgraph.LineageSubgraph:
        """Queries lineage subgraph of this context.

        Returns:
            lineage subgraph(gca_lineage_subgraph.LineageSubgraph): Lineage subgraph of this Context.
        """

        return self.api_client.query_context_lineage_subgraph(
            context=self.resource_name, retry=base._DEFAULT_RETRY
        )

    def get_executions(self) -> List[execution.Execution]:
        """Returns Executions associated to this context.

        Returns:
            executions (List[Executions]): Executions associated to this context.
        """
        return execution.Execution.list(
            filter=metadata_utils._make_filter_string(in_context=[self.resource_name]),
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )
