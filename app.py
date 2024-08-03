from flask import Flask, render_template, request, jsonify, send_file
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import sounddevice as sd
from werkzeug.utils import secure_filename
import soundfile as sf
from pydub import AudioSegment
import msvcrt
from datetime import datetime
import nltk

app = Flask(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
checkpoint = r"models"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model_2 = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model_2.to(device)

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/microauto/page')
def microauto():
    return render_template('index.html')

@app.route('/samsum/page')
def samsum():
    return render_template('index2.html')

@app.route('/samsum', methods=['POST'])
def translate():
    if 'text' not in request.json:
        return jsonify({'error': 'Không có văn bản được cung cấp'}), 400
    text = request.json['text']
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    length = 0
    chunk = ""
    for sentence in sentences:
        combined_length = len(tokenizer.tokenize(sentence)) + length
        if combined_length <= tokenizer.max_len_single_sentence:
            chunk += sentence + " "
            length = combined_length
        else:
            chunks.append(chunk.strip())
            length = len(tokenizer.tokenize(sentence))
            chunk = " " + sentence + " "
            length = len(tokenizer.tokenize(sentence))
    if chunk:
        chunks.append(chunk.strip())
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for chunk in chunks:
        input_encoding = tokenizer(chunk, return_tensors="pt").to(device)
        output = model_2.generate(**input_encoding)
        translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(translated_text)
    return jsonify({'translations': results})

@app.route('/download')
def download_file():
    file_path = 'files/txt/micro_result.txt'
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, port=2468, host='0.0.0.0', ssl_context=('pem/cert.pem', 'pem/key.pem'))
