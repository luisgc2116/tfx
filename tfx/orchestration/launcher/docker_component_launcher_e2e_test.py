# Copyright 2019 Google LLC. All Rights Reserved.
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
"""E2E Tests for tfx.orchestration.launcher.docker_component_launcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_dag_runner
from tfx.orchestration.config import docker_component_config
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.launcher import docker_component_launcher
from tfx.types import component_spec
from tfx.types import standard_artifacts


class _HelloWorldSpec(component_spec.ComponentSpec):
  INPUTS = {}
  OUTPUTS = {}
  PARAMETERS = {
      'name': component_spec.ExecutionParameter(type=str),
  }


class _HelloWorldComponent(base_component.BaseComponent):

  SPEC_CLASS = _HelloWorldSpec
  EXECUTOR_SPEC = executor_spec.ExecutorContainerSpec(
      # TODO(b/143965964): move the image to private repo if the test is flaky
      # due to docker hub.
      image='alpine:latest',
      command=['echo'],
      args=['hello {{exec_properties.name}}'])

  def __init__(self, name):
    super(_HelloWorldComponent, self).__init__(_HelloWorldSpec(name=name))


class _GrepComponentSpec(component_spec.ComponentSpec):
  INPUTS = {
      'input1':
          component_spec.ChannelParameter(
              type=standard_artifacts.ExternalArtifact),
  }
  OUTPUTS = {
      'output1':
          component_spec.ChannelParameter(
              type=standard_artifacts.ExternalArtifact),
  }
  PARAMETERS = {
      'pattern': component_spec.ExecutionParameter(type=str),
  }


class _GrepComponent(base_component.BaseComponent):
  SPEC_CLASS = _GrepComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorContainerSpec(
      image='alpine:latest',
      command=['sh', '-c', 'grep "$2" <"$0" > "$1"'],
      args=[
          '/tmp/inputs/input1/data',
          '/tmp/inputs/output1/data',
          '{{exec_properties.pattern}}',
      ],
      input_path_uris={
          '/tmp/inputs/input1/data': '{{input_dict["input1"][0].uri}}',
      },
      output_path_uris={
          '/tmp/inputs/output1/data': '{{output_dict["output1"][0].uri}}',
      },
  )

  def __init__(self, input1, output1, pattern):
    super(_GrepComponent, self).__init__(
        _GrepComponentSpec(input1=input1, output1=output1, pattern=pattern))


# TODO(hongyes): Add more complicated samples to pass inputs/outputs between
# containers.
def _create_pipeline(
    pipeline_name,
    pipeline_root,
    metadata_path,
    name,
):
  hello_world = _HelloWorldComponent(name=name)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[hello_world],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      additional_pipeline_args={},
  )


def _create_component_io_pipeline(
    pipeline_name,
    pipeline_root,
    metadata_path,
    pattern,
    input1_uri,
    output1_uri,
):
  input1_artifact = standard_artifacts.ExternalArtifact()
  input1_artifact.uri = input1_uri
  output1_artifact = standard_artifacts.ExternalArtifact()
  output1_artifact.uri = output1_uri

  grep_task = _GrepComponent(
      input1=input1_artifact,
      output1=output1_artifact,
      pattern=pattern,
  )

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[grep_task],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      additional_pipeline_args={},
  )


class DockerComponentLauncherE2eTest(tf.test.TestCase):

  def setUp(self):
    super(DockerComponentLauncherE2eTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'docker_e2e_test'
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def testDockerComponentLauncherInBeam(self):

    beam_dag_runner.BeamDagRunner(
        config=pipeline_config.PipelineConfig(
            supported_launcher_classes=[
                docker_component_launcher.DockerComponentLauncher
            ],
            default_component_configs=[
                docker_component_config.DockerComponentConfig()
            ])).run(
                _create_pipeline(
                    pipeline_name=self._pipeline_name,
                    pipeline_root=self._pipeline_root,
                    metadata_path=self._metadata_path,
                    name='docker_e2e_test_in_beam'))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      self.assertEqual(1, len(m.store.get_executions()))

  def test_launching_component_with_io(self):
    input1_uri = os.path.join(self._pipeline_root, 'input1.txt')
    output1_uri = os.path.join(self._pipeline_root, 'output1.txt')

    tf.io.gfile.makedirs(self._pipeline_root)
    with tf.io.gfile.GFile(input1_uri, 'w') as f:
      for i in range(20):
        f.write(str(i))

    beam_dag_runner.BeamDagRunner(
        config=pipeline_config.PipelineConfig(
            supported_launcher_classes=[
                docker_component_launcher.DockerComponentLauncher
            ],
            default_component_configs=[
                docker_component_config.DockerComponentConfig()
            ])).run(
                _create_component_io_pipeline(
                    pipeline_name=self._pipeline_name,
                    pipeline_root=self._pipeline_root,
                    metadata_path=self._metadata_path,
                    input1_uri=input1_uri,
                    output1_uri=output1_uri,
                    pattern='7',
                ))

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      self.assertEqual(1, len(m.store.get_executions()))

    self.assertTrue(tf.io.gfile.exists(output1_uri))


if __name__ == '__main__':
  tf.test.main()
