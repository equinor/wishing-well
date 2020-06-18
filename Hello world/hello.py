from flask import Flask

PORT = 5000

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello world!"

if __name__ == "__main__":
    app.run(port=PORT, debug=True, host="0.0.0.0")
