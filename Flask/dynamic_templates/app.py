from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def home():
    Users = {
                "Archie":"Amsterdam", 
                "Veronica":"London", 
                "Betty":"San Francisco", 
                "Jughead":"Los Angeles"
            }
    return render_template("index.html", users=Users)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)