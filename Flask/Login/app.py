from flask import Flask, render_template
from forms import LoginForm

from APIKEY import apikey

app = Flask(__name__)
app.config['SECRET_KEY'] = apikey

users = {
    "archie.andrews@email.com": "football4life",
    "veronica.lodge@email.com": "fashiondiva"
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        for u_email, u_password in users.items():
            print("Email:", form.email.data)
            print("Password:", form.password.data)
            if u_email == form.email.data and u_password == form.password.data:
                return render_template("login.html", message ="Successfully Logged In")
        return render_template("login.html", form = form, message ="Incorrect Email or Password")
    elif form.errors:
        print(form.errors.items())
    return render_template("login.html", form = form)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)