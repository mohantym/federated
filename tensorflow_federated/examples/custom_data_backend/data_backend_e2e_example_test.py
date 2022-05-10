# Copyright 2022, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.proto.v0 import computation_pb2 as pb


def get_ctx(data_backend):

  def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
    return tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(device), data_backend=data_backend)

  factory = tff.framework.local_executor_factory(leaf_executor_fn=ex_fn)
  return tff.framework.ExecutionContext(executor_fn=factory)


def get_data_handle(uris, element_type):
  element_type_proto = tff.framework.serialize_type(element_type)
  arguments = [
      pb.Computation(data=pb.Data(uri=uri), type=element_type_proto)
      for uri in uris
  ]
  return tff.framework.DataDescriptor(
      None, arguments, tff.FederatedType(element_type, tff.CLIENTS),
      len(arguments))


class DataBackendE2EExampleTest(tff.test.TestCase, parameterized.TestCase):

  def test_constant_data_backend(self):

    class ConstantDataBackend(tff.framework.DataBackend):

      async def materialize(self, data, type_spec):
        return tf.cast(int(data.uri[-1]), tf.int32)

    tff.framework.set_default_context(get_ctx(ConstantDataBackend()))

    element_type = tff.types.TensorType(tf.int32)
    uris = [f'uri://{i}' for i in range(3)]
    data_handle = get_data_handle(uris, element_type)

    @tff.federated_computation(tff.types.FederatedType(tf.int32, tff.CLIENTS))
    def foo(x):
      return tff.federated_sum(x)

    self.assertEqual(foo(data_handle), 3)

  def test_list_data_backend(self):

    class ListDataBackend(tff.framework.DataBackend):

      async def materialize(self, data, type_spec):
        return [1, 2, 3]

    tff.framework.set_default_context(get_ctx(ListDataBackend()))

    element_type = tff.types.SequenceType(tf.int32)
    uris = [f'uri://{i}' for i in range(3)]
    data_handle = get_data_handle(uris, element_type)

    @tff.federated_computation(
        tff.types.FederatedType(element_type, tff.CLIENTS))
    def foo(x):

      @tff.tf_computation(element_type)
      def local_sum(nums):
        return nums.reduce(0, lambda x, y: x + y)

      return tff.federated_sum(tff.federated_map(local_sum, x))

    self.assertEqual(foo(data_handle), 18)

  def test_numpy_arr_data_backend(self):

    class NumpyArrDataBackend(tff.framework.DataBackend):

      async def materialize(self, data, type_spec):
        return np.array([1, 2, 3])

    tff.framework.set_default_context(get_ctx(NumpyArrDataBackend()))

    element_type = tff.TensorType(dtype=tf.int32, shape=[3])
    uris = [f'uri://{i}' for i in range(3)]
    data_handle = get_data_handle(uris, element_type)

    @tff.federated_computation(
        tff.types.FederatedType(element_type, tff.CLIENTS))
    def foo(x):

      @tff.tf_computation(element_type)
      def local_sum(nums):
        return tf.math.reduce_sum(nums)

      return tff.federated_sum(tff.federated_map(local_sum, x))

    self.assertEqual(foo(data_handle), 18)


if __name__ == '__main__':
  tff.test.main()
