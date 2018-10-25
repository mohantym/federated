# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for type_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.impl import test_utils
from tensorflow_federated.python.core.impl import type_utils

from tensorflow_federated.python.core.impl.anonymous_tuple import AnonymousTuple


class TypeUtilsTest(tf.test.TestCase):

  def test_infer_type_with_none(self):
    self.assertEqual(type_utils.infer_type(None), None)

  def test_infer_type_with_scalar_int_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.constant(1))), 'int32')

  def test_infer_type_with_scalar_bool_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.constant(False))), 'bool')

  def test_infer_type_with_int_array_tensor(self):
    self.assertEqual(
        str(type_utils.infer_type(tf.constant([10, 20]))), 'int32[2]')

  def test_infer_type_with_scalar_int_variable_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.Variable(10))), 'int32')

  def test_infer_type_with_scalar_bool_variable_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.Variable(True))), 'bool')

  def test_infer_type_with_scalar_float_variable_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.Variable(0.5))), 'float32')

  def test_infer_type_with_scalar_int_array_variable_tensor(self):
    self.assertEqual(str(type_utils.infer_type(tf.Variable([10]))), 'int32[1]')

  def test_infer_type_with_int_dataset(self):
    self.assertEqual(
        str(type_utils.infer_type(tf.data.Dataset.from_tensors(10))), 'int32*')

  def test_infer_type_with_dict_dataset(self):
    self.assertIn(
        str(type_utils.infer_type(
            tf.data.Dataset.from_tensors({'a': 10, 'b': 20}))),
        ['<a=int32,b=int32>*', '<b=int32,a=int32>*'])

  def test_infer_type_with_int(self):
    self.assertEqual(str(type_utils.infer_type(10)), 'int32')

  def test_infer_type_with_float(self):
    self.assertEqual(str(type_utils.infer_type(0.5)), 'float32')

  def test_infer_type_with_bool(self):
    self.assertEqual(str(type_utils.infer_type(True)), 'bool')

  def test_infer_type_with_string(self):
    self.assertEqual(str(type_utils.infer_type('abc')), 'string')

  def test_infer_type_with_unicode_string(self):
    self.assertEqual(str(type_utils.infer_type(u'abc')), 'string')

  def test_infer_type_with_numpy_int_array(self):
    self.assertEqual(str(type_utils.infer_type(np.array([10, 20]))), 'int64[2]')

  def test_infer_type_with_numpy_float64_scalar(self):
    self.assertEqual(str(type_utils.infer_type(np.float64(1))), 'float64')

  def test_infer_type_with_int_list(self):
    self.assertEqual(str(type_utils.infer_type([1, 2, 3])), 'int32[3]')

  def test_infer_type_with_nested_float_list(self):
    self.assertEqual(
        str(type_utils.infer_type([[0.1], [0.2], [0.3]])), 'float32[3,1]')

  def test_infer_type_with_anonymous_tuple(self):
    self.assertEqual(
        str(type_utils.infer_type(AnonymousTuple([('a', 10), (None, False)]))),
        '<a=int32,bool>')

  def test_infer_type_with_nested_anonymous_tuple(self):
    self.assertEqual(
        str(type_utils.infer_type(AnonymousTuple([
            ('a', 10), (None, AnonymousTuple([(None, True), (None, 0.5)]))]))),
        '<a=int32,<bool,float32>>')

  def test_infer_type_with_namedtuple(self):
    self.assertEqual(
        str(type_utils.infer_type(collections.namedtuple('_', 'y x')(1, True))),
        '<y=int32,x=bool>')

  def test_infer_type_with_dict(self):
    self.assertIn(
        str(type_utils.infer_type({'a': 1, 'b': 2.0})),
        ['<a=int32,b=float32>', '<b=float32,a=int32>'])

  def test_infer_type_with_ordered_dict(self):
    self.assertEqual(
        str(type_utils.infer_type(
            collections.OrderedDict([('b', 2.0), ('a', 1)]))),
        '<b=float32,a=int32>')

  def test_infer_type_with_dataset_list(self):
    self.assertEqual(
        str(type_utils.infer_type([
            tf.data.Dataset.from_tensors(x) for x in [1, True, [0.5]]])),
        '<int32*,bool*,float32[1]*>')

  def test_infer_type_with_nested_dataset_list_tuple(self):
    self.assertEqual(
        str(type_utils.infer_type(tuple([
            (tf.data.Dataset.from_tensors(x),) for x in [1, True, [0.5]]]))),
        '<<int32*>,<bool*>,<float32[1]*>>')

  def test_tf_dtypes_and_shapes_to_type_with_int(self):
    self.assertEqual(
        str(type_utils.tf_dtypes_and_shapes_to_type(
            tf.int32, tf.TensorShape([]))),
        'int32')

  def test_tf_dtypes_and_shapes_to_type_with_int_vector(self):
    self.assertEqual(
        str(type_utils.tf_dtypes_and_shapes_to_type(
            tf.int32, tf.TensorShape([2]))),
        'int32[2]')

  def test_tf_dtypes_and_shapes_to_type_with_dict(self):
    self.assertIn(
        str(type_utils.tf_dtypes_and_shapes_to_type(
            {'a': tf.int32, 'b': tf.bool},
            {'a': tf.TensorShape([]), 'b': tf.TensorShape([5])})),
        ['<a=int32,b=bool[5]>', '<b=bool[5],a=int32>'])

  def test_tf_dtypes_and_shapes_to_type_with_ordered_dict(self):
    self.assertEqual(
        str(type_utils.tf_dtypes_and_shapes_to_type(
            collections.OrderedDict([('b', tf.int32), ('a', tf.bool)]),
            collections.OrderedDict([
                ('b', tf.TensorShape([1])), ('a', tf.TensorShape([]))]))),
        '<b=int32[1],a=bool>')

  def test_tf_dtypes_and_shapes_to_type_with_tuple(self):
    self.assertEqual(
        str(type_utils.tf_dtypes_and_shapes_to_type(
            (tf.int32, tf.bool), (tf.TensorShape([1]), tf.TensorShape([2])))),
        '<int32[1],bool[2]>')

  def test_tf_dtypes_and_shapes_to_type_with_list(self):
    self.assertEqual(
        str(type_utils.tf_dtypes_and_shapes_to_type(
            [tf.int32, tf.bool], [tf.TensorShape([1]), tf.TensorShape([2])])),
        '<int32[1],bool[2]>')

  def test_tf_dtypes_and_shapes_to_type_with_namedtuple(self):
    foo = collections.namedtuple('_', 'y x')
    self.assertEqual(
        str(type_utils.tf_dtypes_and_shapes_to_type(
            foo(x=tf.int32, y=tf.bool),
            foo(x=tf.TensorShape([1]), y=tf.TensorShape([2])))),
        '<y=bool[2],x=int32[1]>')

  def test_tf_dtypes_and_shapes_to_type_with_three_level_nesting(self):
    foo = collections.namedtuple('_', 'y x')
    self.assertEqual(
        str(type_utils.tf_dtypes_and_shapes_to_type(
            foo(x=[tf.int32, {'bar': tf.float32}], y=tf.bool),
            foo(x=[tf.TensorShape([1]), {'bar': tf.TensorShape([2])}],
                y=tf.TensorShape([3])))),
        '<y=bool[3],x=<int32[1],<bar=float32[2]>>>')

  def test_type_to_tf_dtypes_and_shapes_with_int_scalar(self):
    dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes(tf.int32)
    test_utils.assert_nested_struct_eq(dtypes, tf.int32)
    test_utils.assert_nested_struct_eq(shapes, tf.TensorShape([]))

  def test_type_to_tf_dtypes_and_shapes_with_int_vector(self):
    dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes((tf.int32, [10]))
    test_utils.assert_nested_struct_eq(dtypes, tf.int32)
    test_utils.assert_nested_struct_eq(shapes, tf.TensorShape([10]))

  def test_type_to_tf_dtypes_and_shapes_with_tensor_triple(self):
    dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes(
        [('a', (tf.int32, [5])), ('b', tf.bool), ('c', (tf.float32, [3]))])
    test_utils.assert_nested_struct_eq(
        dtypes, {'a': tf.int32, 'b': tf.bool, 'c': tf.float32})
    test_utils.assert_nested_struct_eq(
        shapes, {'a': tf.TensorShape([5]),
                 'b': tf.TensorShape([]),
                 'c': tf.TensorShape([3])})

  def test_type_to_tf_dtypes_and_shapes_with_two_level_tuple(self):
    dtypes, shapes = type_utils.type_to_tf_dtypes_and_shapes(
        [('a', tf.bool), ('b', [('c', tf.float32), ('d', (tf.int32, [20]))])])
    test_utils.assert_nested_struct_eq(
        dtypes, {'a': tf.bool, 'b': {'c': tf.float32, 'd': tf.int32}})
    test_utils.assert_nested_struct_eq(
        shapes, {'a': tf.TensorShape([]),
                 'b': {'c': tf.TensorShape([]), 'd': tf.TensorShape([20])}})


if __name__ == '__main__':
  tf.test.main()
