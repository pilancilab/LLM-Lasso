# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Feature selection experiments."""

import json
import os
import pathlib
import random

from absl import app
from absl import flags
import numpy as np
from llm_lasso.baselines.sequential_attention.sequential_attention.experiments.datasets.dataset import get_dataset
from llm_lasso.baselines.sequential_attention.sequential_attention.experiments.models.mlp_lly import LiaoLattyYangModel
from llm_lasso.baselines.sequential_attention.sequential_attention.experiments.models.mlp_omp import OrthogonalMatchingPursuitModel
from llm_lasso.baselines.sequential_attention.sequential_attention.experiments.models.mlp_sa import SequentialAttentionModel
from llm_lasso.baselines.sequential_attention.sequential_attention.experiments.models.mlp_seql import SequentialLASSOModel
from llm_lasso.baselines.sequential_attention.sequential_attention.experiments.models.mlp_sparse import SparseModel
import tensorflow as tf


os.environ["TF_DETERMINISTIC_OPS"] = "1"

FLAGS = flags.FLAGS

# Experiment parameters
flags.DEFINE_integer("seed", 2023, "Random seed")
flags.DEFINE_enum(
    "data_name",
    "mnist",
    ["mnist", "fashion", "isolet", "mice", "coil", "activity", "gene_cancer"],
    "Data name",
)
flags.DEFINE_string(
    "model_dir",
    "./model_dir",
    "Checkpoint directory for feature selection model",
)

# flags.DEFINE_string("x_train_path", None, "Path to X train CSV")
# flags.DEFINE_string("x_test_path", None, "Path to X test CSV")
# flags.DEFINE_string("y_train_path", None, "Path to y train CSV")
# flags.DEFINE_string("y_test_path", None, "Path to y test CSV")
flags.DEFINE_string("split_dir", None, "Directory with x/y train/test CSVs for each split")
flags.DEFINE_integer("num_splits", 1, "Number of train/test splits to run")


# Feature selection hyperparameters
flags.DEFINE_integer(
    "num_selected_features", 50, "Number of features to select"
)
flags.DEFINE_enum("algo", "sa", ["sa", "lly", "seql", "gl", "omp"], "Algorithm")
flags.DEFINE_integer(
    "num_inputs_to_select_per_step", 1, "Number of features to select at a time"
)

# Hyperparameters
flags.DEFINE_float(
    "val_ratio", 0.125, "How much of the training data to split for validation."
)
flags.DEFINE_list("deep_layers", "67", "Layers in MLP model")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_integer("decay_steps", 250, "Decay steps")
flags.DEFINE_float("decay_rate", 1.0, "Decay rate")
flags.DEFINE_float("alpha", 0, "Leaky ReLU alpha")
flags.DEFINE_bool("enable_batch_norm", False, "Enable batch norm")
flags.DEFINE_float("group_lasso_scale", 0.01, "Group LASSO scale")

# Finer control if needed
flags.DEFINE_integer("num_epochs_select", -1, "Number of epochs to fit")
flags.DEFINE_integer("num_epochs_fit", -1, "Number of epochs to select")


ALGOS = {
    "sa": SequentialAttentionModel,
    "lly": LiaoLattyYangModel,
    "seql": SequentialLASSOModel,
    "gl": SequentialLASSOModel,
    "omp": OrthogonalMatchingPursuitModel,
}


def run_trial(batch_size=256,
              num_epochs_select=250,
              num_epochs_fit=250,
              learning_rate=0.0002,
              decay_steps=100,
              decay_rate=1.0,
              x_train_path=None,
              x_test_path=None,
              y_train_path=None,
              y_test_path=None,
              split_index=None):
  """Run Sequential-Attention (or other algo) on one split."""

  # ---------- 1. Load the dataset ----------
  datasets = get_dataset(
      FLAGS.data_name,
      FLAGS.val_ratio,
      batch_size,
      x_train_path=x_train_path,
      x_test_path=x_test_path,
      y_train_path=y_train_path,
      y_test_path=y_test_path,
  )
  ds_train, ds_val, ds_test = datasets["ds_train"], datasets["ds_val"], datasets["ds_test"]
  is_classification   = datasets["is_classification"]
  num_classes         = datasets["num_classes"]
  num_features        = datasets["num_features"]
  num_train_steps_sel = num_epochs_select * len(ds_train)

  loss_fn = (tf.keras.losses.CategoricalCrossentropy()
             if is_classification else tf.keras.losses.MeanAbsoluteError())

  # ---------- 2. Build model/FS arguments ----------
  mlp_args = dict(
      layer_sequence=[int(i) for i in FLAGS.deep_layers],
      is_classification=is_classification,
      num_classes=num_classes,
      learning_rate=learning_rate,
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      alpha=FLAGS.alpha,
      batch_norm=FLAGS.enable_batch_norm,
  )
  fs_args = dict(
      num_inputs=num_features,
      num_inputs_to_select=FLAGS.num_selected_features,
  )

  if FLAGS.algo == "sa":
    fs_args.update(num_inputs_to_select_per_step=FLAGS.num_inputs_to_select_per_step,
                   num_train_steps=num_train_steps_sel)
  if FLAGS.algo in ("seql", "omp"):
    fs_args["num_train_steps"] = num_train_steps_sel
  if FLAGS.algo in ("seql", "gl"):
    fs_args["group_lasso_scale"] = FLAGS.group_lasso_scale
  if FLAGS.algo == "gl":     # GL selects all k at once
    fs_args["num_inputs_to_select_per_step"] = FLAGS.num_selected_features
  if FLAGS.algo == "lly":    # LLY ignores num_inputs_to_select
    fs_args.pop("num_inputs_to_select", None)

  # ---------- 3. Train feature-selector ----------
  print("Selecting features …")
  selector = ALGOS[FLAGS.algo](**mlp_args, **fs_args)
  selector.compile(loss=loss_fn, metrics=["accuracy"])
  selector.fit(ds_train, validation_data=ds_val,
               epochs=num_epochs_select, verbose=2)

  # ---------- 4. Extract selected indices ----------
  if   FLAGS.algo == "sa":
      sel_vec = selector.seqatt.selected_features
      _, selected_indices = tf.math.top_k(sel_vec, k=FLAGS.num_selected_features)
      selected_indices = selected_indices.numpy()
  elif FLAGS.algo == "lly":
      x_train = datasets["x_train"]
      logits  = selector.lly(tf.convert_to_tensor(x_train))
      _, selected_indices = tf.math.top_k(logits, k=FLAGS.num_selected_features)
      selected_indices = selected_indices.numpy()
  elif FLAGS.algo in ("gl", "seql"):
      selected_indices = selector.seql.selected_features_history.numpy().tolist()
  elif FLAGS.algo == "omp":
      selected_indices = selector.omp.selected_features_history.numpy().tolist()

  assert len(selected_indices) == FLAGS.num_selected_features

  # ---------- 5. Persist selected feature list ----------
  split_tag = f"_{split_index+1}" if split_index is not None else ""
  save_txt  = os.path.join(FLAGS.split_dir, f"{FLAGS.algo}_selected{split_tag}.txt")
  with open(save_txt, "w") as fp:
      fp.write(",".join(map(str, selected_indices)))
  print(f"Saved selected features → {save_txt}")

  # ---------- 6. Sparse re-training (optional) ----------
  one_hot = tf.math.reduce_sum(
      tf.one_hot(selected_indices, num_features, dtype=tf.float32), axis=0)

  re_model = SparseModel(selected_features=one_hot, **mlp_args)
  re_model.compile(loss=loss_fn, metrics=["accuracy"])
  re_model.fit(ds_train, validation_data=ds_val,
               epochs=num_epochs_fit, verbose=2)

  # ---------- 7. Eval ----------
  res_val  = re_model.evaluate(ds_val,  return_dict=True)["accuracy"]
  res_test = re_model.evaluate(ds_test, return_dict=True)["accuracy"]
  print(f"Split {split_index}: val={res_val:.4f}, test={res_test:.4f}")

  # (optionally write JSON to model_dir/fit/results.json … left unchanged)

  return {"val_acc": res_val, "test_acc": res_test, "indices": selected_indices}


def main(args):
  del args  # Not used.

  os.environ["PYTHONHASHSEED"] = str(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)

  tf.keras.backend.clear_session()

  num_epochs_select = FLAGS.num_epochs
  num_epochs_fit = FLAGS.num_epochs
  if FLAGS.num_epochs_select > 0:
    num_epochs_select = FLAGS.num_epochs_select
  if FLAGS.num_epochs_fit > 0:
    num_epochs_fit = FLAGS.num_epochs_fit
  if FLAGS.data_name == "gene_cancer" and FLAGS.split_dir:
    for i in range(FLAGS.num_splits):
      x_train_path = os.path.join(FLAGS.split_dir, f"x_train{i}.csv")
      x_test_path = os.path.join(FLAGS.split_dir, f"x_test{i}.csv")
      y_train_path = os.path.join(FLAGS.split_dir, f"y_train{i}.csv")
      y_test_path = os.path.join(FLAGS.split_dir, f"y_test{i}.csv")

      print(f"\n====== Running Split {i} ======")

      results = run_trial(
        batch_size=FLAGS.batch_size,
        num_epochs_select=num_epochs_select,
        num_epochs_fit=num_epochs_fit,
        learning_rate=FLAGS.learning_rate,
        decay_steps=FLAGS.decay_steps,
        decay_rate=FLAGS.decay_rate,
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        split_index=i,  # we'll use this below to save results
      )

  else:
    run_trial(
        batch_size=FLAGS.batch_size,
        num_epochs_select=num_epochs_select,
        num_epochs_fit=num_epochs_fit,
        learning_rate=FLAGS.learning_rate,
        decay_steps=FLAGS.decay_steps,
        decay_rate=FLAGS.decay_rate,
    )

if __name__ == "__main__":
  app.run(main)
