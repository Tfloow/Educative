from flask import Flask, render_template, abort, session, redirect, url_for
from forms import LoginForm, SignUpForm

from APIKEY import apikey

app = Flask(__name__)

app.config["SECRET_KEY"] = apikey

"""Information regarding the Pets in the System."""
pets = [
            {"id": 1, "name": "Nelly", "age": "5 weeks", "bio": "I am a tiny kitten rescued by the good people at Paws Rescue Center. I love squeaky toys and cuddles."},
            {"id": 2, "name": "Yuki", "age": "8 months", "bio": "I am a handsome gentle-cat. I like to dress up in bow ties."},
            {"id": 3, "name": "Basker", "age": "1 year", "bio": "I love barking. But, I love my friends more."},
            {"id": 4, "name": "Mr. Furrkins", "age": "5 years", "bio": "Probably napping."}, 
        ]

"""Information regarding the Users in the System."""
users = [
            {"id": 1, "full_name": "Pet Rescue Team", "email": "team@pawsrescue.co", "password": "adminpass"},
        ]

@app.route("/")
def home():
    return render_template("home.html", my_pets=pets)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/details/<pet_id>")
def details(pet_id):
    for pet in pets:
        if pet["id"] == int(pet_id):
            return render_template("details.html", my_pet=pet)
    abort(404, description="No such ID in our shelter")
    
@app.route("/sign-up", methods=["GET", "POST"])
def sign_up():
    form = SignUpForm()
    
    if form.validate_on_submit():
        print(form.email.data)
        print(form.Name.data)
        print(form.password.data)
        return render_template("login.html", form=form, result="Success")
    if form.is_submitted():
        return render_template("login.html", form=form, result="Failure")
    if form.errors:
        print(form.errors.items())
    
    return render_template("signup.html", form=form, result="")

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    
    if form.validate_on_submit():
        # If the form is correctly submitted
        for user in users:
            if user["email"] == form.email.data and user["password"] == form.password.data:
                print("successful login")
                print(session)
                session['user'] = user
                print(session)
                return render_template("login.html", message="Successful Login")
        return render_template("login.html", form=form, message="Wrong login")
    
    return render_template("login.html", form=form)

@app.route("/logout")
def logout():
    if 'user' in session:
        session.pop('user')
    return redirect(url_for('home', _scheme='https', _external=True))
# use http if debugging locally


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)