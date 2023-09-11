## Variables

- [Variables](#variables)
  - [Scope and Shadowing](#scope-and-shadowing)


A variable is like a box. We store data inside a variable and the variable has a name. The name of a variable is called *identifier*. By default they are all **immutable**.

```rust
let language ;
```

We just instantiate a variable.

```rust
let language = "Rust";
```

We just created a variable and assigned a string to it.

#### Mutable Variable

To make a variable mutable we just need to add the identifier `mut`:

```rust
let mut language = "Rust";
```

We can use tuple like to assign multiple variable:

```rust
let (course,category) = ("Rust","beginner");
```

### Scope and Shadowing

FOr the scope it's like in C where it depends of the codeblock. We can redeclare a variable outside and the scope of the variable will change. We call this **shadowing**. So when a variable that takes the same name in the inner block as that of variable in the outer block. This concept is called