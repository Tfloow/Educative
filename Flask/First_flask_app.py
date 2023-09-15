from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/<my_var>")
def name(my_var):
    return "Hi " + my_var + " 😋"

if __name__ == "__main__":
    app.run(debug=True)