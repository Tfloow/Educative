# Learn Rust from Scratch

- [Learn Rust from Scratch](#learn-rust-from-scratch)
  - [Getting Started](#getting-started)
    - [The Basic Program](#the-basic-program)
    - [Basic Formatting](#basic-formatting)
  - [Printing Styles](#printing-styles)
  - [Comments](#comments)
  - [Variables](#variables)
    - [Scope and Shadowing](#scope-and-shadowing)
  - [Data Types](#data-types)
    - [Numeric Types](#numeric-types)
    - [Boolean](#boolean)
    - [Character and String](#character-and-string)
    - [Arrays](#arrays)
    - [Tuples](#tuples)
    - [Constant Variables](#constant-variables)
  - [Operators](#operators)
    - [Arithmetic](#arithmetic)
    - [Logical](#logical)
    - [Comparison](#comparison)
    - [Bitwise](#bitwise)
    - [Type Casting Operator](#type-casting-operator)
    - [Borrowing and Dereferencing Operators](#borrowing-and-dereferencing-operators)
  - [Conditional Expressions](#conditional-expressions)
    - [If](#if)
    - [If Let](#if-let)
    - [Match](#match)
  - [Loops](#loops)


## Getting Started

### The Basic Program

We wrote a little program ([here](hello-rust/src/main.rs))

```rust
fn main() {
    println!("Hello World!");
}
```

First we instantiate the main function that is actually gonna run `fn main()`.

We print everything with `println!()`. Like in C we need to end with a semi-colon.

We also define a code block like in C with `{}`.

We see we added `!` before `println` which is a **macro**. Macro is used in *metaprogramming*. In other words, it is code that writes code. It is not a function that call like functions. They are expanded and provide more run-time features. (more info [here](https://doc.rust-lang.org/book/ch19-06-macros.html))

### Basic Formatting

We can use a placeholder like in python:
```rust
println!("{}", 1)
```

We can use formatting to print variables value. We can also use multiple placeholders like:

```rust
fn main(){
    println!("Hello my name is {} and I am {}", "John", "20" );
}
```

We can also use *positional* arguments which is just indicating what goes where with `{0}` or `{1}` etc.

We can also *name an argument* like this:

```rust
fn main() {
    println!("My name is  {name} and I do {job}", name="John", job="Science");
}
```

We can also format data to have a certain way to display.

```{:b},{:x},{:o}```
 
 b is for binary, x is for hexadecimal then o is octal.

 We can also compute inside of those brackets.

 We can also use the bracket as a debug thanks to the `{:?}` and it helps us printing multiple value like:

 ```rust
 fn main() {
    println!("{:?}", ("This is a Rust Course", 101));
}
```

## Printing Styles

We have different type of printing:
![Print type](image.png)

## Comments

We have different types of comment:
1. Line comments: `//`
2. Block comments: `/*...*/`
3. Doc comments: `///` and `//!`
   1. Outer doc comments: `///` They are written outside of the code block (it supports markdown notation)
   2. Inner doc comments: `//!` So we add the comment inside of a code block.

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

FOr the scope it's like in C where it depends of the codeblock. We can redeclare a variable outside and the scope of the variable will change. We call this **shadowing**. So when a variable that takes the same name in the inner block as that of variable in the outer block.


## Data Types

- [Learn Rust from Scratch](#learn-rust-from-scratch)
  - [Getting Started](#getting-started)
    - [The Basic Program](#the-basic-program)
    - [Basic Formatting](#basic-formatting)
  - [Printing Styles](#printing-styles)
  - [Comments](#comments)
  - [Variables](#variables)
    - [Scope and Shadowing](#scope-and-shadowing)
  - [Data Types](#data-types)
    - [Numeric Types](#numeric-types)
    - [Boolean](#boolean)
    - [Character and String](#character-and-string)
    - [Arrays](#arrays)
    - [Tuples](#tuples)
    - [Constant Variables](#constant-variables)
  - [Operators](#operators)
    - [Arithmetic](#arithmetic)
    - [Logical](#logical)
    - [Comparison](#comparison)
    - [Bitwise](#bitwise)
    - [Type Casting Operator](#type-casting-operator)
    - [Borrowing and Dereferencing Operators](#borrowing-and-dereferencing-operators)
  - [Conditional Expressions](#conditional-expressions)
    - [If](#if)
    - [If Let](#if-let)
    - [Match](#match)
  - [Loops](#loops)


Rust is a **statically typed** language so we need to specify the data type at compile time.

We can define the variable like:

```rust
let variable name = value;
```

So we don't really specify the datatype. The compiler know what the datatype will be.

```rust
let variable name:datatype = value;
```

In Rust we have some *primitive* datatype.

![Chart from the lesson on educative](image-1.png)

### Numeric Types

#### Integer

So when we want to specify the datatype, we need to either choose a *signed* or *unsigned*. So signed is an integer that can be positive or negative. Then we specify the amount of bits with `8`, `16`, `32` or `64`.

```rust
fn main() {
    //explicitly define an integer
    let a:i32 = 24;
}
```

#### Floating Point 

We have also float (so number like `3.141`). We can specify it with `f` and then we can choose between `32` or `64`.

### Boolean

It is either a `true` or `false`. This datatype `bool`.

### Character and String

Character is just one character as it says. We need to use single quote. So something `let my_char = 'a'`. The datatype is `char`.

#### String

A string is multiple character defined with a *double quote*. `let program = "Rust"`. We use `&str` for the datatype. (if you know C prior you know why there is a `&`).

### Arrays

Do define an array we need to do `let name: [type; size] = [elem1, elem2, elem3];`. So we need to specify what we put in the array and what is the fixed amount in the array. We can also instantiate an array with the same data with a certain amount.

```rust
fn main() {
   //define an array of size 4
   let arr:[i32;4] = [1, 2, 3, 4]; 
   // initialize an array of size 4 with 0
   let arr1 = [0 ; 4]; 
   print!("{:?}", arr1);
}
```

We can make an array mutable by adding the keyword `mut` to change the values.

To print the whole array we need to print with the debug option `{:?}`.

We can get the length of the array with `arr1.len()`. 

#### Slice

We can slice array with `let slice_array2:&[i32] = &arr[0..2];`.

### Tuples

There is two way to write a tuple:

#### 1. 

We can do tuple with multiple datatype.

```rust
let tupleName = ("Rust", 'a', 1);
```

#### 2.

We can specify the datatype of each element in the tuple.

```rust
let tupleName:(&str, char, i32) = ("Rust", 'a', 1);
```

To access to data we need to do something like `tupleName.0`.

As with array, we can print it with debug and add `mut` to make it mutable.

### Constant Variables

We can set a variable constant with `const` keyword.

The difference between `const` and `let`. We can declare the constant variable as *global* with `const` we cannot do this with `let`.

We must define the datatype for a constant. We also need the `const` to be set before running so we cannot store the result at runtime:

```rust
let a = *a random variable*; // ✔
const b = *a random variable*; // ✖
```

We cannot do shadowing with constant.

## Operators

- [Learn Rust from Scratch](#learn-rust-from-scratch)
  - [Getting Started](#getting-started)
    - [The Basic Program](#the-basic-program)
    - [Basic Formatting](#basic-formatting)
  - [Printing Styles](#printing-styles)
  - [Comments](#comments)
  - [Variables](#variables)
    - [Scope and Shadowing](#scope-and-shadowing)
  - [Data Types](#data-types)
    - [Numeric Types](#numeric-types)
    - [Boolean](#boolean)
    - [Character and String](#character-and-string)
    - [Arrays](#arrays)
    - [Tuples](#tuples)
    - [Constant Variables](#constant-variables)
  - [Operators](#operators)
    - [Arithmetic](#arithmetic)
    - [Logical](#logical)
    - [Comparison](#comparison)
    - [Bitwise](#bitwise)
    - [Type Casting Operator](#type-casting-operator)
    - [Borrowing and Dereferencing Operators](#borrowing-and-dereferencing-operators)
  - [Conditional Expressions](#conditional-expressions)
    - [If](#if)
    - [If Let](#if-let)
    - [Match](#match)
  - [Loops](#loops)

![Alt text](image-2.png)
![Alt text](image-3.png)

### Arithmetic

![Alt text](image-4.png)

### Logical

![Alt text](image-5.png)

### Comparison

![Alt text](image-6.png)

### Bitwise

![Alt text](image-7.png)

### Type Casting Operator

When we want to convert some datatype into another we need to call the `as` keyword like so:

```rust
let a:i32 = 10;
let b = (a as f32)/2.0;
```

### Borrowing and Dereferencing Operators

References are just like *pointer* in C. For example we can save where something points:

```rust
let operand1 = & operand2
let operand3 = &mut operand2
```

The first is a *shared* borrow. So we can read data of ``operand2``. The second is **mutable** so we can read and modify data.


```rust
fn main() {
    let x = 10;
    let mut y = 13;
    //immutable reference to a variable
    let a = &x;
    println!("Value of a:{}", a); 
    println!("Value of x:{}", x); // x value remains the same since it is immutably borrowed
    //mutable reference to a variable
    let b = &mut y;
    println!("Value of b:{}", b);

    *b = 11; // derefencing 
    println!("Value of b:{}", b); // updated value of b
    println!("Value of y:{}", y); // y value can be changed as it is mutuably borrowed
}
```

Output:
```
Value of a:10
Value of x:10
Value of b:13
Value of b:11
Value of y:11
```

To modify the value inside of a pointer we need to like in C so we use a `*`.

## Conditional Expressions 

In Rust we have 3 types of conditional expressions:

1. `if`
2. `if let`
3. `match`

### If

It's simply `if condition` and then a codeblock. We can also have `else`.


The short version of this is something like:

```rust
let x = if(condition) { statement } else { statement };
```

### If Let

It is a conditional with some *pattern matching*.

```rust
if let(value 1, value 2) = match_expression{
    // Do something
}
```

We can also let Rust guess the last value like this:

```rust
fn main() {
    // define a scrutinee expression    
    let course = ("Rust", "beginner","course");
    // pattern matches with the scrutinee expression
    if let ("Rust", "beginner", c) = course {
        println!("Wrote first two values in pattern to be matched with the scrutinee expression : {}", c);
    } 
    else {
        // do not execute this block
        println!("Value unmatched");
    }
}
```

### Match 

It lets us check if a value is part of a list of value.

```rust
fn main() {
    // define a variable 
    let x = 5;
    // define match expression
    match x {
        1 => println!("Java"),
        2 => println!("Python"),
        3 => println!("C++"),
        4 => println!("C#"),
        5 => println!("Rust"),
        6 => println!("Kotlin"),
        _ => println!("Some other value"),
    };
}
```

So thanks to this match function we can easily set the value of a variable depending of another value. like this:

```rust
fn main(){
   // define a variable
   let course = "Rust";
   // return value of match expression in a variable
   let found_course = match course {
      "Rust" => "Rust",
      "Java" => "Java",
      "C++" => "C Plus Plus",
      "C#" => "C Sharp",
      _ => "Unknown Language"
   };
   println!("Course name : {}",found_course);
}
```

## Loops


