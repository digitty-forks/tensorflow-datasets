# coding=utf-8
# Copyright 2025 The TensorFlow Datasets Authors.
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

import json
import os
import tempfile
from typing import Optional
from unittest import mock

from absl.testing import parameterized
import apache_beam as beam
from etils import epath
import tensorflow as tf
from tensorflow_datasets import testing
from tensorflow_datasets.core import dataset_utils
from tensorflow_datasets.core import example_parser
from tensorflow_datasets.core import file_adapters
from tensorflow_datasets.core import naming
from tensorflow_datasets.core import writer as writer_lib
from tensorflow_datasets.core.utils import shard_utils


class GetShardSpecsTest(testing.TestCase):
  # Here we don't need to test all possible reading configs, as this is tested
  # by shard_utils.py.

  def test_1bucket_6shards(self):
    specs = writer_lib._get_shard_specs(
        num_examples=8,
        total_size=16,
        bucket_lengths=[8],
        filename_template=naming.ShardedFileTemplate(
            dataset_name='bar',
            split='train',
            data_dir='/',
            filetype_suffix='tfrecord',
        ),
        shard_config=shard_utils.ShardConfig(num_shards=6),
    )
    self.assertEqual(
        specs,
        [
            # Shard#, path, from_bucket, examples_number, reading instructions.
            writer_lib._ShardSpec(
                0,
                '/bar-train.tfrecord-00000-of-00006',
                '/bar-train.tfrecord-00000-of-00006_index.json',
                1,
                [
                    shard_utils.FileInstruction(
                        filename='0', skip=0, take=1, examples_in_shard=8
                    ),
                ],
            ),
            writer_lib._ShardSpec(
                1,
                '/bar-train.tfrecord-00001-of-00006',
                '/bar-train.tfrecord-00001-of-00006_index.json',
                2,
                [
                    shard_utils.FileInstruction(
                        filename='0', skip=1, take=2, examples_in_shard=8
                    ),
                ],
            ),
            writer_lib._ShardSpec(
                2,
                '/bar-train.tfrecord-00002-of-00006',
                '/bar-train.tfrecord-00002-of-00006_index.json',
                1,
                [
                    shard_utils.FileInstruction(
                        filename='0', skip=3, take=1, examples_in_shard=8
                    ),
                ],
            ),
            writer_lib._ShardSpec(
                3,
                '/bar-train.tfrecord-00003-of-00006',
                '/bar-train.tfrecord-00003-of-00006_index.json',
                1,
                [
                    shard_utils.FileInstruction(
                        filename='0', skip=4, take=1, examples_in_shard=8
                    ),
                ],
            ),
            writer_lib._ShardSpec(
                4,
                '/bar-train.tfrecord-00004-of-00006',
                '/bar-train.tfrecord-00004-of-00006_index.json',
                2,
                [
                    shard_utils.FileInstruction(
                        filename='0', skip=5, take=2, examples_in_shard=8
                    ),
                ],
            ),
            writer_lib._ShardSpec(
                5,
                '/bar-train.tfrecord-00005-of-00006',
                '/bar-train.tfrecord-00005-of-00006_index.json',
                1,
                [
                    shard_utils.FileInstruction(
                        filename='0', skip=7, take=1, examples_in_shard=8
                    ),
                ],
            ),
        ],
    )

  def test_4buckets_2shards(self):
    specs = writer_lib._get_shard_specs(
        num_examples=8,
        total_size=16,
        bucket_lengths=[2, 3, 0, 3],
        filename_template=naming.ShardedFileTemplate(
            dataset_name='bar',
            split='train',
            data_dir='/',
            filetype_suffix='tfrecord',
        ),
        shard_config=shard_utils.ShardConfig(num_shards=2),
    )
    self.assertEqual(
        specs,
        [
            # Shard#, path, examples_number, reading instructions.
            writer_lib._ShardSpec(
                0,
                '/bar-train.tfrecord-00000-of-00002',
                '/bar-train.tfrecord-00000-of-00002_index.json',
                4,
                [
                    shard_utils.FileInstruction(
                        filename='0', skip=0, take=2, examples_in_shard=2
                    ),
                    shard_utils.FileInstruction(
                        filename='1', skip=0, take=2, examples_in_shard=3
                    ),
                ],
            ),
            writer_lib._ShardSpec(
                1,
                '/bar-train.tfrecord-00001-of-00002',
                '/bar-train.tfrecord-00001-of-00002_index.json',
                4,
                [
                    shard_utils.FileInstruction(
                        filename='1', skip=2, take=-1, examples_in_shard=3
                    ),
                    shard_utils.FileInstruction(
                        filename='3', skip=0, take=-1, examples_in_shard=3
                    ),
                ],
            ),
        ],
    )


def _read_records(path, file_format=file_adapters.DEFAULT_FILE_FORMAT):
  """Returns (files_names, list_of_records_in_each_file).

  Args:
    path: path to tfrecord, omitting suffix.
    file_format: format of the record files.
  """
  # Ignore _index.json files.
  paths = sorted(tf.io.gfile.glob('%s-*-of-*' % path))
  paths = [p for p in paths if not p.endswith(writer_lib._INDEX_PATH_SUFFIX)]
  all_recs = []
  for fpath in paths:
    all_recs.append(
        list(
            dataset_utils.as_numpy(
                file_adapters.ADAPTER_FOR_FORMAT[file_format].make_tf_data(
                    fpath
                )
            )
        )
    )
  return [os.path.basename(p) for p in paths], all_recs


def _read_indices(path):
  """Returns (files_name, list of index in each file).

  Args:
    path: path to index, omitting suffix.
  """
  paths = sorted(tf.io.gfile.glob('%s-*-of-*_index.json' % path))
  all_indices = []
  for path in paths:
    json_str = epath.Path(path).read_text()
    # parse it back into a proto.
    shard_index = json.loads(json_str)
    all_indices.append(list(shard_index['index']))
  return [os.path.basename(p) for p in paths], all_indices


class WriterTest(testing.TestCase):
  EMPTY_SPLIT_ERROR = 'No examples were yielded.'
  TOO_SMALL_SPLIT_ERROR = 'num_examples (1) < number_of_shards (2)'

  NUM_SHARDS = 5
  RECORDS_TO_WRITE = [
      (1, b'a'),
      (2, b'b'),
      (3, b'c'),
      (4, b'd'),
      (5, b'e'),
      (6, b'f'),
      (7, b'g'),
      (8, b'hi'),
  ]
  RECORDS_WITH_HOLES = [
      (1, b'a'),
      (2, b'b'),
      (3, b'c'),
      (4, b'd'),
      (50000, b'e'),
      (600000, b'f'),
      (7000000, b'g'),
      (80000000, b'hi'),
  ]
  SHARDS_CONTENT = [
      [b'f', b'g'],
      [b'd'],
      [b'a', b'b'],
      [b'hi'],
      [b'e', b'c'],
  ]
  SHARDS_CONTENT_NO_SHUFFLING = [
      [b'a', b'b'],
      [b'c'],
      [b'd', b'e'],
      [b'f'],
      [b'g', b'hi'],
  ]

  def _write(
      self,
      to_write,
      salt: str = '',
      dataset_name: str = 'foo',
      split: str = 'train',
      disable_shuffling: bool = False,
      ignore_duplicates: bool = False,
      example_writer: Optional[writer_lib.ExampleWriter] = None,
      shard_config: Optional[shard_utils.ShardConfig] = None,
  ):
    example_writer = example_writer or writer_lib.ExampleWriter(
        file_format=file_adapters.DEFAULT_FILE_FORMAT
    )
    filetype_suffix = file_adapters.ADAPTER_FOR_FORMAT[
        example_writer.file_format
    ].FILE_SUFFIX
    filename_template = naming.ShardedFileTemplate(
        dataset_name=dataset_name,
        split=split,
        filetype_suffix=filetype_suffix,
        data_dir=self.tmp_dir,
    )
    shard_config = shard_config or shard_utils.ShardConfig(
        num_shards=self.NUM_SHARDS
    )
    writer = writer_lib.Writer(
        serializer=testing.DummySerializer('dummy specs'),
        filename_template=filename_template,
        hash_salt=salt,
        disable_shuffling=disable_shuffling,
        example_writer=example_writer,
        shard_config=shard_config,
        ignore_duplicates=ignore_duplicates,
    )
    for key, record in to_write:
      writer.write(key, record)
    return writer.finalize()

  def test_write_tfrecord(self):
    """Stores records as tfrecord in a fixed number of shards with shuffling."""
    path = os.path.join(self.tmp_dir, 'foo-train.tfrecord')
    shards_length, total_size = self._write(to_write=self.RECORDS_TO_WRITE)
    self.assertLen(shards_length, self.NUM_SHARDS)
    self.assertEqual(
        shards_length, [len(shard) for shard in self.SHARDS_CONTENT]
    )
    self.assertEqual(total_size, 9)
    written_files, all_recs = _read_records(path)
    written_index_files, all_indices = _read_indices(path)
    self.assertEqual(
        written_files,
        [
            f'foo-train.tfrecord-{i:05d}-of-{self.NUM_SHARDS:05d}'
            for i in range(self.NUM_SHARDS)
            if shards_length[i]
        ],
    )
    self.assertEqual(all_recs, self.SHARDS_CONTENT)
    self.assertEmpty(written_index_files)
    self.assertEmpty(all_indices)

  def test_write_tfrecord_sorted_by_key(self):
    """Stores records as tfrecord in a fixed number of shards without shuffling."""
    path = os.path.join(self.tmp_dir, 'foo-train.tfrecord')
    shards_length, total_size = self._write(
        to_write=self.RECORDS_TO_WRITE, disable_shuffling=True
    )
    self.assertEqual(
        shards_length,
        [len(shard) for shard in self.SHARDS_CONTENT_NO_SHUFFLING],
    )
    self.assertEqual(total_size, 9)
    written_files, all_recs = _read_records(path)
    written_index_files, all_indices = _read_indices(path)
    self.assertEqual(
        written_files,
        [
            f'foo-train.tfrecord-{i:05d}-of-{self.NUM_SHARDS:05d}'
            for i in range(self.NUM_SHARDS)
            if shards_length[i]
        ],
    )
    self.assertEqual(all_recs, self.SHARDS_CONTENT_NO_SHUFFLING)
    self.assertEmpty(written_index_files)
    self.assertEmpty(all_indices)

  def test_write_tfrecord_sorted_by_key_with_holes(self):
    """Stores records as tfrecord in a fixed number of shards without shuffling."""
    path = os.path.join(self.tmp_dir, 'foo-train.tfrecord')
    shards_length, total_size = self._write(
        to_write=self.RECORDS_WITH_HOLES, disable_shuffling=True
    )
    self.assertEqual(
        shards_length,
        [len(shard) for shard in self.SHARDS_CONTENT_NO_SHUFFLING],
    )
    self.assertEqual(total_size, 9)
    written_files, all_recs = _read_records(path)
    written_index_files, all_indices = _read_indices(path)
    self.assertEqual(
        written_files,
        [
            f'foo-train.tfrecord-{i:05d}-of-{self.NUM_SHARDS:05d}'
            for i in range(self.NUM_SHARDS)
            if shards_length[i]
        ],
    )
    self.assertEqual(all_recs, self.SHARDS_CONTENT_NO_SHUFFLING)
    self.assertEmpty(written_index_files)
    self.assertEmpty(all_indices)

  def test_custom_writer(self):
    custom_example_writer = CustomExampleWriter()
    _, total_size = self._write(
        to_write=self.RECORDS_TO_WRITE, example_writer=custom_example_writer
    )
    self.assertEqual(total_size, 9)
    self.assertEqual(custom_example_writer.num_examples_written, 8)

  @mock.patch.object(example_parser, 'ExampleParser', testing.DummyParser)
  def test_write_duplicated_keys(self):
    to_write = [(1, b'a'), (2, b'b'), (1, b'c')]
    with self.assertRaisesWithPredicateMatch(
        AssertionError, 'Two examples share the same hashed key'
    ):
      shard_config = shard_utils.ShardConfig(num_shards=1)
      self._write(to_write=to_write, shard_config=shard_config)

  @mock.patch.object(example_parser, 'ExampleParser', testing.DummyParser)
  def test_write_duplicated_keys_ignore_duplicates(self):
    to_write = [(1, b'a'), (2, b'b'), (1, b'c')]
    shard_config = shard_utils.ShardConfig(num_shards=1)
    shards_length, total_size = self._write(
        to_write=to_write, shard_config=shard_config, ignore_duplicates=True
    )
    self.assertEqual(shards_length, [2])
    self.assertEqual(total_size, 2)

  def test_empty_split(self):
    to_write = []
    with self.assertRaisesWithPredicateMatch(
        AssertionError, self.EMPTY_SPLIT_ERROR
    ):
      shard_config = shard_utils.ShardConfig(num_shards=1)
      self._write(to_write=to_write, shard_config=shard_config)

  def test_too_small_split(self):
    to_write = [(1, b'a')]
    with self.assertRaisesWithPredicateMatch(
        AssertionError, self.TOO_SMALL_SPLIT_ERROR
    ):
      shard_config = shard_utils.ShardConfig(num_shards=2)
      self._write(to_write=to_write, shard_config=shard_config)
      self._write(to_write=to_write)


def _get_runner() -> beam.runners.PipelineRunner:
  return beam.runners.DirectRunner()


class TfrecordsWriterBeamTest(testing.TestCase):
  NUM_SHARDS = 3
  RECORDS_TO_WRITE = [(i, str(i).encode('utf-8')) for i in range(10)]
  RECORDS_WITH_DUPLICATES = [
      (1, b'a'),
      (1, b'a'),
      (1, b'a'),
      (2, b'b'),
  ]
  SHARDS_CONTENT = [
      [b'6', b'9'],
      [b'7'],
      [b'4', b'1', b'2', b'8', b'0', b'5', b'3'],
  ]
  SHARDS_CONTENT_NO_SHUFFLING = [
      [b'0', b'1', b'2'],
      [b'3', b'4', b'5'],
      [b'6', b'7', b'8', b'9'],
  ]

  def _write(
      self,
      to_write,
      salt: str = '',
      dataset_name: str = 'foo',
      split: str = 'train',
      disable_shuffling: bool = False,
      ignore_duplicates: bool = False,
      example_writer: Optional[writer_lib.ExampleWriter] = None,
      shard_config: Optional[shard_utils.ShardConfig] = None,
  ):
    example_writer = example_writer or writer_lib.ExampleWriter(
        file_format=file_adapters.DEFAULT_FILE_FORMAT
    )
    filetype_suffix = file_adapters.ADAPTER_FOR_FORMAT[
        example_writer.file_format
    ].FILE_SUFFIX
    filename_template = naming.ShardedFileTemplate(
        dataset_name=dataset_name,
        split=split,
        filetype_suffix=filetype_suffix,
        data_dir=self.tmp_dir,
    )
    shard_config = shard_config or shard_utils.ShardConfig(
        num_shards=self.NUM_SHARDS
    )
    writer = writer_lib.BeamWriter(
        serializer=testing.DummySerializer('dummy specs'),
        filename_template=filename_template,
        hash_salt=salt,
        disable_shuffling=disable_shuffling,
        example_writer=example_writer,
        shard_config=shard_config,
        ignore_duplicates=ignore_duplicates,
    )
    # Here we need to disable type check as `beam.Create` is not capable of
    # inferring the type of the PCollection elements.
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=False
    )
    with beam.Pipeline(options=options) as pipeline:

      @beam.ptransform_fn
      def _build_pcollection(pipeline):
        pcollection = pipeline | 'Start' >> beam.Create(to_write)
        return writer.write_from_pcollection(pcollection)

      _ = pipeline | 'test' >> _build_pcollection()  # pylint: disable=no-value-for-parameter
    return writer.finalize()

  def test_write_tfrecord(self):
    """Stores records as tfrecord in a fixed number of shards with shuffling."""
    path = os.path.join(self.tmp_dir, 'foo-train.tfrecord')
    shards_length, total_size = self._write(to_write=self.RECORDS_TO_WRITE)
    self.assertLen(shards_length, self.NUM_SHARDS)
    self.assertEqual(
        shards_length, [len(shard) for shard in self.SHARDS_CONTENT]
    )
    self.assertEqual(total_size, 10)
    written_files, all_recs = _read_records(path)
    written_index_files, all_indices = _read_indices(path)
    self.assertEqual(
        written_files,
        [
            f'foo-train.tfrecord-{i:05d}-of-{self.NUM_SHARDS:05d}'
            for i in range(self.NUM_SHARDS)
            if shards_length[i]
        ],
    )
    self.assertEqual(all_recs, self.SHARDS_CONTENT)
    self.assertEmpty(written_index_files)
    self.assertEmpty(all_indices)

  def test_write_tfrecord_with_duplicates(self):
    with self.assertRaisesWithPredicateMatch(
        AssertionError, 'Two examples share the same hashed key'
    ):
      self._write(to_write=self.RECORDS_WITH_DUPLICATES)

  def test_write_tfrecord_with_ignored_duplicates(self):
    shards_length, total_size = self._write(
        to_write=self.RECORDS_WITH_DUPLICATES, ignore_duplicates=True
    )
    self.assertEqual(shards_length, [2])
    self.assertEqual(total_size, 2)

  def test_empty_split(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        'The total number of generated examples is 0 for split train. This'
        ' should be >0!',
    ):
      self._write(to_write=[])

  def test_write_tfrecord_sorted_by_key(self):
    """Stores records as tfrecord in a fixed number of shards without shuffling."""
    path = os.path.join(self.tmp_dir, 'foo-train.tfrecord')
    shards_length, total_size = self._write(
        to_write=self.RECORDS_TO_WRITE, disable_shuffling=True
    )
    self.assertEqual(
        shards_length,
        [len(shard) for shard in self.SHARDS_CONTENT_NO_SHUFFLING],
    )
    self.assertEqual(total_size, 10)
    written_files, all_recs = _read_records(path)
    written_index_files, all_indices = _read_indices(path)
    self.assertEqual(
        written_files,
        [
            f'foo-train.tfrecord-{i:05d}-of-{self.NUM_SHARDS:05d}'
            for i in range(self.NUM_SHARDS)
            if shards_length[i]
        ],
    )
    self.assertEqual(all_recs, self.SHARDS_CONTENT_NO_SHUFFLING)
    self.assertEmpty(written_index_files)
    self.assertEmpty(all_indices)

  def test_write_tfrecord_sorted_by_key_with_holes(self):
    """Stores records as tfrecord in a fixed number of shards without shuffling.

    Note that the keys are not consecutive but contain gaps.
    """
    path = os.path.join(self.tmp_dir, 'foo-train.tfrecord')
    records_with_holes = [(i**4, str(i**4).encode('utf-8')) for i in range(10)]
    expected_shards = [
        [b'0', b'1', b'16', b'81', b'256', b'625', b'1296'],
        [b'2401', b'4096'],
        [b'6561'],
    ]

    shards_length, total_size = self._write(
        to_write=records_with_holes, disable_shuffling=True
    )
    self.assertEqual(shards_length, [len(shard) for shard in expected_shards])
    self.assertEqual(total_size, 28)
    written_files, all_recs = _read_records(path)
    written_index_files, all_indices = _read_indices(path)
    self.assertEqual(
        written_files,
        [
            f'foo-train.tfrecord-{i:05d}-of-{self.NUM_SHARDS:05d}'
            for i in range(self.NUM_SHARDS)
        ],
    )
    self.assertEqual(all_recs, expected_shards)
    self.assertEmpty(written_index_files)
    self.assertEmpty(all_indices)


class NoShuffleBeamWriterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('tfrecord', file_adapters.FileFormat.TFRECORD),
  )
  def test_write_beam(self, file_format: file_adapters.FileFormat):

    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_dir = epath.Path(tmp_dir)

      def get_writer(split):
        filename_template = naming.ShardedFileTemplate(
            dataset_name='foo',
            split=split,
            filetype_suffix=file_format.file_suffix,
            data_dir=tmp_dir,
        )
        return writer_lib.NoShuffleBeamWriter(
            serializer=testing.DummySerializer('dummy specs'),
            filename_template=filename_template,
            file_format=file_format,
        )

      to_write = [(i, str(i).encode('utf-8')) for i in range(10)]
      # Here we need to disable type check as `beam.Create` is not capable of
      # inferring the type of the PCollection elements.
      options = beam.options.pipeline_options.PipelineOptions(
          pipeline_type_check=False
      )
      writers = [get_writer(split) for split in ('train-b', 'train')]

      for writer in writers:
        with beam.Pipeline(options=options, runner=_get_runner()) as pipeline:

          @beam.ptransform_fn
          def _build_pcollection(pipeline, writer):
            pcollection = pipeline | 'Start' >> beam.Create(to_write)
            return writer.write_from_pcollection(pcollection)

          _ = pipeline | 'test' >> _build_pcollection(writer)

      files = list(tmp_dir.iterdir())
      self.assertGreaterEqual(len(files), 2)
      for f in files:
        self.assertIn(file_format.file_suffix, f.name)
      for writer in writers:
        shard_lengths, total_size = writer.finalize()
        self.assertNotEmpty(shard_lengths)
        self.assertEqual(sum(shard_lengths), 10)
        self.assertGreater(total_size, 10)


class CustomExampleWriter(writer_lib.ExampleWriter):

  def __init__(self):
    super().__init__(file_adapters.FileFormat.TFRECORD)
    self.num_examples_written = 0

  def write(self, path, examples) -> file_adapters.ExamplePositions | None:
    self.num_examples_written += len(list(examples))
    epath.Path(path).touch()


class ExampleWriterTest(parameterized.TestCase):

  def test_multi_output_example_writer(self):
    tfrecord_writer = mock.create_autospec(writer_lib.ExampleWriter)
    tfrecord_writer.file_format = file_adapters.FileFormat.TFRECORD

    riegeli_writer = mock.create_autospec(writer_lib.ExampleWriter)
    riegeli_writer.file_format = file_adapters.FileFormat.RIEGELI

    path = '/tmp/dataset-train.tfrecord-00000-of-00001'
    iterator = [
        ('key1', b'value1'),
        ('key2', b'value2'),
    ]
    writer = writer_lib.MultiOutputExampleWriter([
        tfrecord_writer,
        riegeli_writer,
    ])
    writer.write(path=path, examples=iterator)
    tfrecord_writer.write.assert_called_once_with(path, mock.ANY)
    riegeli_writer.write.assert_called_once_with(
        '/tmp/dataset-train.riegeli-00000-of-00001', mock.ANY
    )


if __name__ == '__main__':
  testing.test_main()
