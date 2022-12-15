from flask import Flask
import pickle
from flask import render_template
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

app.config['SECRET_KEY'] = 'asldfkjlj'


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template("about.html")


# @app.route("/upload", methods=['GET', 'POST'])
# def upload_image():
#     return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
