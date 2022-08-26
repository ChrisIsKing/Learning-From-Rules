import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os,sys
import pickle
from tqdm import tqdm


def sentences_to_elmo_sentence_embs(messages,batch_size=64):
  module_url = "https://tfhub.dev/google/elmo/2"
  elmo = hub.load(module_url)
  message_embeddings = []
  for i in tqdm(range(0,len(messages),batch_size)):
    message_batch = messages[i:i+batch_size]
    embeddings_batch = elmo.signatures['default'](tf.constant(message_batch))['default']
    message_embeddings.extend(embeddings_batch)
  return message_embeddings


