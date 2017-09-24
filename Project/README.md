# Deep Learning Project: Question-Answer Style Chatbot using Deep Learning Sequence to Sequence LSTM model #
- Develop a chatbot using deep recurrent neural network (Sequence to Sequence LSTM) for language models, together with several improvement techniques (Beam Search, Attention Mechanism)
- Develop a web-based user interface for user-chatbot conversation
- Technologies: Python, NLTK, Tensorflow, Flask

I) Overview
=================================================================================================================
1) Requirement:
- Python 3
- nltk
- tensorflow
- flask

2) Description:
- datasets folder: contains 2 conversation datasets: "twitter" and "cornell-movie subtitle"
- models folder: contains any saved models
- data.py: process raw conversation dataset and generate input (question) and output (answer) data
- configuration.py: config for dataset selection, data processing options and model training parameters
- run.py: main code for model training
- seq2seq_model.py: sequence to sequence model (from Tensorflow)
- demo.py: run webapp for chat user interface

II) Instruction
=================================================================================================================
1) Model Training:
- Run: python run.py train
- Trained model will be saved to models folder. Training with GPU Tensorflow will take about 4 hours. Training with CPU only will take 2 days.

2) Chatbot Demo:
- Run: python demo.py
- A web interface will be shown for you to key in question. Flask will call the trained model in the back end to generate answer.




