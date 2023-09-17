from flask import Flask, render_template
from forms import LoginForm
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dfewfew123213rwdsgert34tgfd1234trgf'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
db = SQLAlchemy(app)

users = {
    "archie.andrews@email.com": "football4life",
    "veronica.lodge@email.com": "fashiondiva"
}

class User(db.Model):
    email = db.Column(db.String, primary_key=True, unique=True, nullable=False)
    password = db.Column(db.String, nullable=False)

# Create the application context
with app.app_context():
    # Create the database tables
    db.create_all()


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        for u_email, u_password in users.items():
            if u_email == form.email.data and u_password == form.password.data:
                return render_template("login.html", message ="Successfully Logged In")
        return render_template("login.html", message ="Incorrect Email or Password")
    elif form.errors:
        print(form.errors.items())
    return render_template("login.html", form = form)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)