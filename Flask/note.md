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
    - [Variables](#variables)
    - [Control flow](#control-flow)
    - [Template Inheritance](#template-inheritance)


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

We want something generic on the server side but dynamic on the client side.

We can have dynamic templating thanks to Jinja. Jinja works inside the html file we just need to provide some *delimiters*:
- ``{% ... %}`` is used for statements.
- ``{{ ... }}`` is used for variables.
- ``{# ... #}`` is used for comments.
- ``# ... ##`` is used for line statements.

### Variables

Than to Flask, we can pass any Python object in the template. To pass an object we first need to tell Python we are passing this object by doing:

```python
return render_template("index.html", my_object = Object)
```

and then we can use it with:

```html
{{ my_object }}
```

### Control flow

Jinja provides syntax to handle control flow.

#### Loops

So for example a loop looks like this:

```python
{% for elements in array %}
    ...
{% endfor %}
```

So we can iterate to create more content easily as shown [here](dynamic_templates/app.py).

#### Conditionals

```python
{% if true %}
{% endif %}
```

We can also use ``elif`` and ``else``.

### Template Inheritance

It is useful when two or more pages are similar and rather than just copy pasting we can use their shared content with a `base.html`.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="{{url_for('static', filename='format.css')}}" />
    
  <title>{% block title %}<!-- Placeholder for Title -->{% endblock %} - Jinja Demo</title>
   
    {% block head %} 
    <!-- Placeholder for Other Imports -->
    {% endblock %}
    
</head>
<body>
    <div id="header"> JINJA DEMO </div>
    <div id="content">
        {% block content %}
        <!-- Placeholder for Page Content -->
        {% endblock %}
    </div>
    <div id="footer"> Copyright © 2019 All Rights Reserved </div>
</body>
</html>
```

We jut created our block content (and finished it with the endblock).

In Jinja, we first specify a block by using the keyword `block` then we give it a title (here `content`). More example [here](template_inheritance). As we can see, we also need to use the keyword `extends` to know what is this page needing.
