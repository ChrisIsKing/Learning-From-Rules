import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os,sys
import pickle
from tqdm import tqdm

# tf.compat.v1.disable_eager_execution()

def sentences_to_elmo_sentence_embs(messages,batch_size=64):
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  #message_lengths = [len(m.split()) for m in messages]
  module_url = "https://tfhub.dev/google/elmo/3"
  elmo = hub.load(module_url)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  with tf.compat.v1.Session(config=sess_config) as session:
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    message_embeddings = []
    for i in tqdm(range(0,len(messages),batch_size)):
      #print("Embedding sentences from {} to {}".format(i,min(i+batch_size,len(messages))-1))
      message_batch = messages[i:i+batch_size]
      #length_batch = message_lengths[i:i+batch_size]
      embeddings_batch = elmo.signatures['default'](tf.constant(message_batch))['default']
      # embeddings_batch = session.run(elmo(message_batch,signature="default",as_dict=True))["default"]
      #embeddings_batch = get_embeddings_list(embeddings_batch, length_batch, ELMO_EMBED_SIZE)
      message_embeddings.extend(embeddings_batch)
  return message_embeddings


def sentences_to_elmo_sentence_embs_2(messages,batch_size=64):
  module_url = "https://tfhub.dev/google/elmo/2"
  elmo = hub.load(module_url)
  message_embeddings = []
  for i in tqdm(range(0,len(messages),batch_size)):
    message_batch = messages[i:i+batch_size]
    embeddings_batch = elmo.signatures['default'](tf.constant(message_batch))['default']
    message_embeddings.extend(embeddings_batch)
  return message_embeddings