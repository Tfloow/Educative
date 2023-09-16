# Flask

- [Flask](#flask)
  - [Introduction to Flask](#introduction-to-flask)
    - [Model-Template-View Architecture](#model-template-view-architecture)
    - [WSGI and Jinja2](#wsgi-and-jinja2)
  - [First Flask Application](#first-flask-application)
    - [URL Routes and Views](#url-routes-and-views)
    - [Dynamic Routing](#dynamic-routing)
  - [Static Templates](#static-templates)
  - [Static Files](#static-files)
  - [Dynamic Templates](#dynamic-templates)


## Introduction to Flask

Flask is pretty lightweight by being a *micro-framework*. We will use *http* and *https* communication with `GET`, `POST`, `PUT` and `DELETE`.

### Model-Template-View Architecture

It is a **software architectural pattern** that is divided in 3 components:
1. Models: represents how data is stored in the database.
2. Views: are the components that are visible to the user (GUI/output).
3. Controllers: are the components that act as an interface between models and views. So it connects the user input to the model.

![Nice illustration from Educative describing MVC Architecture](image.png)

Nice illustration from Educative describing MVC Architecture.

### WSGI and Jinja2

**Web Server Gateway Interface** or WSGI is a standard that describes the communication between a web server and a client application. More info in PEP333.

Jinja is a template language used in Python.

Template is what the user sees, the front-end.

## First Flask Application

The code is located [here](First_flask_app.py).

### URL Routes and Views

In Flask, each function we create **must** be bind to a meaningful URL with the `@app.route(path)`.

This decorator takes:
- `rule`: The URL that is passed.
- `endpoint`: The name of the view function.
- `options`: optional parameter.

The last two one are not mandatory.

### Dynamic Routing

We can also make a rule more flexible by making it depend of a variable like this `"/<my_var>"`.

If we want to pass something else than a regular word, we need to use a **converter** like this: `"/<int:number>"`.

We will also all along this course work on [Paws.py](Paws.py).

## Static Templates

A static template is the `HTML` file that remains constant. By essence, HTML is *static*.

We can pass HTML code like `"<h1>Hello welcome</h1>"` to do a title on a page.

But it's not really suited for rendering a whole app. We use `render_template()`. It has 2 arguments:
1. `template_name_or_list`: the name of a template or an iterable list of templates.
2. `context`: optional and variable that should be available inside the template.

So our code looks something like:

```python
def view_name():
    return render_template(template_name)
```

It will look for the template files in a directory called `/templates`.


## Static Files

To store static files or assets, we put all of this in the `/static` directory. Then when we want our webpage to load this content with the correct *endpoint* we need to use `url_for(view_function_name, variable_name = value_of_variable)`.

In our case we need to do `url_for('static', filename = 'name_of_file')`. And this need to be inside the `href` like we can see [here](templates/home.html).

## Dynamic Templates