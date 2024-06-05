import argparse
import sys
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.summary import summary

def import_to_tensorboard(model_dir, log_dir):
  """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.

  Args:
    model_dir: The location of the protobuf (`pb`) model to visualize
    log_dir: The location for the Tensorboard log to begin visualization from.

  Usage:
    Call this function with your model location and desired log directory.
    Launch Tensorboard by pointing it to the log directory.
    View your imported `.pb` model as a graph.
  """

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.saved_model.load(model_dir, tags=['serve'])

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))
    
if __name__ == '__main__':
    """
    Write a summary of the frozen TF graph to a text file.
    Run python inspect_pb.py frozen.pb text_file.txt
    """
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        required=True,
        help="The location of the protobuf (\'pb\') model to visualize.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        required=True,
        help="The location for the Tensorboard log to begin visualization from.")
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=lambda _: import_to_tensorboard(FLAGS.model_dir, FLAGS.log_dir),
            argv=[sys.argv[0]] + unparsed)
