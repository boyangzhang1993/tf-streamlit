from __future__ import absolute_import, division, print_function
# from __future__ import division
# from __future__ import print_function



class BaseHparams(object):
  """Default hyperparameters."""

  def __init__(self,
               total_epochs=100,
               learning_rate=0.004,
               l2=0.001,
               batch_size=20,
               window_size=21,
               ref_path='hs37d5.fa.gz',
               vcf_path='NA12878_calls.vcf.gz',
               bam_path='NA12878_sliced.bam',
               out_dir='examples',
               model_dir='ngs_model',
               log_dir='logs'):

    self.total_epochs = total_epochs
    self.learning_rate = learning_rate
    self.l2 = l2
    self.batch_size = batch_size
    self.window_size = window_size
    self.ref_path = ref_path
    self.vcf_path = vcf_path
    self.bam_path = bam_path
    self.out_dir = out_dir
    self.model_dir = model_dir
    self.log_dir = log_dir

hparams = BaseHparams(batch_size=200)

def get_dataset(hparams, filename, num_epochs):
  """Reads in and processes the TFRecords dataset.

  Builds a pipeline that returns pairs of features, label.
  """

  # Define field names, types, and sizes for TFRecords.
  proto_features = {
      'A_counts':
          tf.io.FixedLenFeature(shape=[hparams.window_size], dtype=tf.float32),
      'C_counts':
          tf.io.FixedLenFeature(shape=[hparams.window_size], dtype=tf.float32),
      'G_counts':
          tf.io.FixedLenFeature(shape=[hparams.window_size], dtype=tf.float32),
      'T_counts':
          tf.io.FixedLenFeature(shape=[hparams.window_size], dtype=tf.float32),
      'ref_sequence':
          tf.io.FixedLenFeature(shape=[hparams.window_size], dtype=tf.int64),
      'label':
          tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
  }

  def _process_input(proto_string):
    """Helper function for input function that parses a serialized example."""

    parsed_features = tf.io.parse_single_example(
        serialized=proto_string, features=proto_features)

    # Stack counts/fractions for all bases to create input of dimensions
    # `hparams.window_size` x len(_ALLOWED_BASES).
    feature_columns = []
    for base in _ALLOWED_BASES:
      feature_columns.append(parsed_features['%s_counts' % base])
    features = tf.stack(feature_columns, axis=-1)
    label = parsed_features['label']
    return features, label

  ds = tf.data.TFRecordDataset(filenames=filename)
  for raw_record in ds.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
  ds = ds.map(map_func=_process_input)

  ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
  ds = ds.batch(batch_size=hparams.batch_size).repeat(count=num_epochs)
  return ds
import os
import random
# import numpy as np


import tensorflow as tf
from tensorflow import keras

cnn_model = tf.keras.models.load_model('saved_model')
stringlist = []
cnn_model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
_ALLOWED_BASES = 'ACGT'

_TRAIN = 'train.tfrecord'
_EVAL = 'eval.tfrecord'
_TEST = 'test.tfrecord'
a_c_g_t_dict = {0:"A", 1: "C", 2:'G', 3:"T"}

# Input dataset
eval_dataset = get_dataset(hparams, filename=os.path.join(hparams.out_dir, _EVAL), num_epochs=1)
# evalution = cnn_model.evaluate(eval_dataset, verbose=1)


# Take one element (inputs, reference_atgc, reference)
def take_one_element(dataset_tensor):
    raw_1 = dataset_tensor.take(1)
    element = tf.data.Dataset.get_single_element(raw_1)
    
    reference = []
    reference_decode = []

    reference_code = element[1]
    reference_code_ref = reference_code.numpy()

    for code in reference_code_ref: 
        reference.append(a_c_g_t_dict[code[0]])
        reference_decode.append(code[0])
    print("".join(reference))
    return element, "".join(reference), reference

element, reference_string, reference_atgc = take_one_element(eval_dataset)


# Make prediction based on element[0] and CNN
# Return prediction_string, prediction_atgc

def prediction_code(element):
    predict_1 = cnn_model.predict(element[0], verbose=1)
    result = []
    for base in predict_1:
        max_pro = 0
        for index, value in enumerate(base):
            if value > max_pro:
                max_pro = value
                max_index = index
        result.append(a_c_g_t_dict[max_index])
    return "".join(result), result

prediction_string_cnn, prediction_atgc_cnn = prediction_code(element)

def prediction_arg_max(element):
    predict_2 = tf.argmax(element[0], axis = 2)
    predict_2 = predict_2.numpy()
    result = []
    for array in predict_2:
        
        result.append(a_c_g_t_dict[array[10]])
    return "".join(result), result




prediction_string_argmax, prediction_atgc_argmax = prediction_arg_max(element)

print(f"reference:     {reference_string}")

print(f"CNN prediction:{prediction_string_cnn}")
# print(f"Softmax prediction:{prediction_string_argmax}")

def check_accuracy(ref, prediction):
    total = len(ref)-1
    r_index = 0 
    missed = 0 
    while r_index <= total:
        if ref[r_index] != prediction[r_index]:
            missed += 1
        r_index += 1
    return 100*missed/ total
error_rate = check_accuracy(reference_atgc, prediction_atgc_cnn)
print(f"CNN error rate: {check_accuracy(reference_atgc, prediction_atgc_cnn)}")
# print(f"Softmax error rate: {check_accuracy(reference_atgc, prediction_atgc_argmax)}")        
def run_one_example(batch_size_run):
    hparams = BaseHparams(batch_size = batch_size_run)
    eval_dataset = get_dataset(hparams, filename=os.path.join(hparams.out_dir, _EVAL), num_epochs=1)

def check_string(ref, prediction):
    total = len(ref)-1
    r_index = 0 
    missed = 0 
    result_reference = []
    result_prediction = []
    next_line = 0
    while r_index <= total:
        if ref[r_index] != prediction[r_index]:
            result_reference.append(f"<span class='highlight blue'>{ref[r_index]}</span> ")
            result_prediction.append(f"<span class='highlight red'>{prediction[r_index]}</span> ")
        else:
            if next_line // 10:
                result_reference.append(f"{ref[r_index]} \n")
                result_prediction.append(f"{prediction[r_index]} \n")
            else:
                result_reference.append(f"{ref[r_index]}")
                result_prediction.append(f"{prediction[r_index]}")
        r_index += 1
        next_line += 1
    return "".join(result_reference), "".join(result_prediction)


result_reference, result_prediction = check_string(reference_atgc, prediction_atgc_cnn)





# decoded = tf.argmax(predict_1, axis=1)
# print(decoded)
# print(reference_decode)



import streamlit as st


from load_css import local_css

local_css("style.css")
 
st.title('DNA Sequencing Error Corrections using Tensorflow')
t = "<h2>Background:<h2>"
st.markdown(t, unsafe_allow_html=True)
st.write("A team in Google showed how Nucleus can be used alongside TensorFlow for solving machine learning problems in genomics [link](https://blog.tensorflow.org/2019/01/using-nucleus-and-tensorflow-for-dna.html). This App is to show you its prediction accuracy with one example")

t = "<h2>After training: <h2>"


st.markdown(t, unsafe_allow_html=True)
st.write("The TensorFlow model can make DNA Sequencing error corrections")

t = "<h3>1. Summary of the TensorFlow model:<h3>"
st.markdown(t, unsafe_allow_html=True)
st.write(stringlist)

t = "<h3>2. One input:<h3>"
st.markdown(t, unsafe_allow_html=True)
st.write("One input contains a tensor with shape (21,4) because of one-hot encoding and 21 windows size.")
st.write(tf.strings.as_string(element[0][1], precision=2, shortest=True))

t = "<h3>3. Predictions:<h3>"
st.markdown(t, unsafe_allow_html=True)
st.write("Based on the input, the mode predict one predicted DNA base.")
st.write("The below is a comparsion between a reference sequence and the predicted sequence: ")


# t = "<div>Hello there my <span class='highlight blue'>name </span> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"

# st.markdown(t, unsafe_allow_html=True)

# with st.container():
#     # st.write("This is inside the container")

#     # You can call any Streamlit command, including custom components:
#     # ('This is a success message!')
#     t2 = result_reference
#     st.markdown(t2, unsafe_allow_html=True)

#     t2 = result_prediction
#     st.markdown(t2, unsafe_allow_html=True)

# t = f"<h3>4. The error rates of this test: {error_rate:.2f} % <h3>"
# st.markdown(t, unsafe_allow_html=True)

def one_run(eval_dataset, force_error = False):
    if force_error:
        error_rate = 0
        while error_rate == 0:
            element, reference_string, reference_atgc = take_one_element(eval_dataset)
            prediction_string_cnn, prediction_atgc_cnn = prediction_code(element)
            error_rate = check_accuracy(reference_atgc, prediction_atgc_cnn)
            result_reference, result_prediction = check_string(reference_atgc, prediction_atgc_cnn)
    else:
        element, reference_string, reference_atgc = take_one_element(eval_dataset)
        prediction_string_cnn, prediction_atgc_cnn = prediction_code(element)
        error_rate = check_accuracy(reference_atgc, prediction_atgc_cnn)
        result_reference, result_prediction = check_string(reference_atgc, prediction_atgc_cnn)
  

    return result_reference, result_prediction, error_rate


import streamlit as st
import pandas as pd


if 'num' not in st.session_state:
    st.session_state.num = 1
if 'data' not in st.session_state:
    st.session_state.data = []


class NewPrediction:
    def __init__(self, page_id, reference, prediction, error_rate):
        
        self.reference = reference
        self.prediction = prediction
        self.error_rate = error_rate

        # self.name = st.markdown(reference, unsafe_allow_html=True)
    

def main():
    placeholder = st.empty()
    placeholder2 = st.empty()

    while True:    
        num = st.session_state.num

        if placeholder2.button('Show overall error rates', key=num):
            # placeholder2.empty()
            df = pd.DataFrame(st.session_state.data)
            st.dataframe(df)
            break
        else:        
            with placeholder.form(key=str(num)):
                st.title(f"One run")
                one_result = one_run(eval_dataset, force_error = True)
                new_student = NewPrediction(page_id=1, reference=one_result[0],  prediction = one_result[1], error_rate = one_result[2])        
                t2 = new_student.reference
                st.write('Reference sequence')
                st.markdown(t2, unsafe_allow_html=True)

                t2 = new_student.prediction
                st.write('**Predicted sequence**')
                st.markdown(t2, unsafe_allow_html=True)
                st.write('**Error rate (%):**')
                
                st.markdown(round(new_student.error_rate, 3), unsafe_allow_html=True)

                if st.form_submit_button('Run 10 times'):
                    for i in range(1, 11):
                        one_result = one_run(eval_dataset, force_error = False)
                        new_student = NewPrediction(page_id=1, reference=one_result[0],  prediction = one_result[1], error_rate = one_result[2])
                        st.session_state.data.append({'error rate': new_student.error_rate})
                        st.session_state.num += 1
                    placeholder.empty()
                    placeholder2.empty()
                else:
                    st.stop()

main()

