# strong_rc

This library is an implementation of **strong-only** reference counted smart pointers in Rust. All applicable methods have identical names as in [`std::rc`], and so this
can be used as a drop in replacement for that.

[![Build Status](https://travis-ci.org/Centril/rust_strong_rc.svg?branch=master)](https://travis-ci.org/Centril/rust_strong_rc)
[![Build status](https://ci.appveyor.com/api/projects/status/1k62lx84j5oflynu?svg=true)](https://ci.appveyor.com/project/Centril/rust_strong_rc)
[![Crates.io](https://img.shields.io/crates/v/strong_rc.svg?maxAge=2592000)](https://crates.io/crates/strong_rc)

[Documentation](https://docs.rs/strong_rc)

## Usage

First, add this to your `Cargo.toml`:

```toml
[dependencies]
strong_rc = "0.1.0"
```

Next, add this to your crate:

```rust
extern crate strong_rc;
use strong_rc::Rc;
```

Then you should consult the documentation, or simply use it as if it were
[`std::rc`].

# License

`strong_rc` is distributed under the terms of both the MIT license and
the Apache License (Version 2.0).

<!-- references -->

[`std::rc`]: https://doc.rust-lang.org/std/rc/index.html

<!-- references -->