from flask import Flask, render_template, request
from flask import jsonify

app = Flask(__name__,static_url_path="/static")

#############
# Routing
#
@app.route('/message', methods=['POST'])
def reply():
    return jsonify( { 'text': generate(request.form['msg'] ) } )

@app.route("/")
def index():
    return render_template("index.html")
#############

'''
Call Seq2seq model to process question and return reply
'''
#_________________________________________________________________

import seq2seq_model
import data
import numpy as np
import configurations

# data
data_path = configurations.data_path
metadata, idx_q, idx_a = data.load_data(PATH=data_path)

# parameters 
source_vocab_size = configurations.vocab_size + 2 # 2 extra UNK and _ (for pad)
target_vocab_size = source_vocab_size
source_len = configurations.source_len 
target_len = configurations.target_len
batch_size = configurations.batch_size
layer_size = configurations.layer_size
num_layers = configurations.num_layers
model_type = configurations.model_type
attention_heads = configurations.attention_heads
model_path = configurations.model_path

model = seq2seq_model.Seq2Seq(source_len=source_len,
                           target_len=target_len,
                           source_vocab_size=source_vocab_size,
                           target_vocab_size=target_vocab_size,
                           model_path=model_path,
                           layer_size=layer_size,
                           num_layers=num_layers,
                           model_type = model_type,
                           attention_heads = attention_heads
                           )
sess = model.load_model()


def generate(question):
    q_indices = data.encode(sequence=question, lookup=metadata['w2idx'], separator=' ')
    idx_q = np.zeros([1, data.limit['maxq']], dtype=np.int32)
    idx_q[0] = np.array(q_indices)
    idx_q = idx_q.T
            
    idx_a = model.predict(sess, idx_q)
    answer = data.decode(sequence=idx_a[0], lookup=metadata['idx2w'], separator=' ')
    return answer

#_________________________________________________________________

# start app
if (__name__ == "__main__"):
    app.run(port = 5001)
