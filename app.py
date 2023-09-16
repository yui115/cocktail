import os
from flask import (
     Flask, 
     request, 
     render_template)

from model import recommend

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        taste = request.form['taste']
        base = request.form['base']
        word = request.form['word']
        name, alc, feature, img_path, url = recommend(base,taste,word)
        return render_template('result.html', name=name, alc=alc, feature=feature, img_path=img_path, url=url)

if __name__ == "__main__":
    app.run(debug=True)