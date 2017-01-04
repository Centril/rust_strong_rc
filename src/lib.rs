// Copyright 2013-2017 The Rust Project Developers, Mazdak Farrokhzad. See
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Single-threaded reference-counting pointers.
//!
//! **Unlike [`std::rc`], this is a strong-only `Rc` meant to be used by caches.
//! The design of this is identical in every respect to [`std::rc`], except for
//! the omission of [`Weak`][weak].
//! Therefore, it can be used as a drop-in replacement.**
//!
//! The type [`Rc<T>`][rc] provides shared ownership of a value of type `T`,
//! allocated in the heap. Invoking [`clone`][clone] on `Rc` produces a new
//! pointer to the same value in the heap. When the last `Rc` pointer to a
//! given value is destroyed, the pointed-to value is also destroyed.
//!
//! Shared references in Rust disallow mutation by default, and `Rc` is no
//! exception. If you need to mutate through an `Rc`, use [`Cell`][cell] or
//! [`RefCell`][refcell].
//!
//! `Rc` uses non-atomic reference counting. This means that overhead is very
//! low, but an `Rc` cannot be sent between threads, and consequently `Rc`
//! does not implement [`Send`][send]. As a result, the Rust compiler
//! will check *at compile time* that you are not sending `Rc`s between
//! threads. If you need multi-threaded, atomic reference counting, use
//! [`sync::Arc`][arc].
//!
//! A cycle between `Rc` pointers will never be deallocated.
//! If you need cycles, use the `Rc` from [`std::rc`].
//!
//! `Rc<T>` automatically dereferences to `T` (via the [`Deref`][deref] trait),
//! so you can call `T`'s methods on a value of type `Rc<T>`. To avoid name
//! clashes with `T`'s methods, the methods of `Rc<T>` itself are [associated
//! functions][assoc], called using function-like syntax:
//!
//! ```
//! use strong_rc::Rc;
//! let my_rc = Rc::new(());
//!
//! assert_eq!(Rc::strong_count(&my_rc), 1);
//! ```
//!
//! [rc]: struct.Rc.html
//! [weak]: struct.Weak.html
//! [`std::rc`]: https://doc.rust-lang.org/std/rc
//! [clone]: https://doc.rust-lang.org/std/clone/trait.Clone.html#tymethod.clone
//! [cell]: https://doc.rust-lang.org/std/cell/struct.Cell.html
//! [refcell]: https://doc.rust-lang.org/std/cell/struct.RefCell.html
//! [send]: https://doc.rust-lang.org/std/marker/trait.Send.html
//! [arc]: https://doc.rust-lang.org/std/sync/struct.Arc.html
//! [deref]: https://doc.rust-lang.org/std/ops/trait.Deref.html
//! [option]: https://doc.rust-lang.org/std/option/enum.Option.html
//! [assoc]: https://doc.rust-lang.org/book/method-syntax.html#associated-functions
//!
//! # Examples
//!
//! Consider a scenario where a set of `Gadget`s are owned by a given `Owner`.
//! We want to have our `Gadget`s point to their `Owner`. We can't do this with
//! unique ownership, because more than one gadget may belong to the same
//! `Owner`. `Rc` allows us to share an `Owner` between multiple `Gadget`s,
//! and have the `Owner` remain allocated as long as any `Gadget` points at it.
//!
//! ```
//! use strong_rc::Rc;
//!
//! struct Owner {
//!     name: String,
//!     // ...other fields
//! }
//!
//! struct Gadget {
//!     id: i32,
//!     owner: Rc<Owner>,
//!     // ...other fields
//! }
//!
//! fn main() {
//!     // Create a reference-counted `Owner`.
//!     let gadget_owner: Rc<Owner> = Rc::new(
//!         Owner {
//!             name: "Gadget Man".to_string(),
//!         }
//!     );
//!
//!     // Create `Gadget`s belonging to `gadget_owner`. Cloning the `Rc<Owner>`
//!     // value gives us a new pointer to the same `Owner` value, incrementing
//!     // the reference count in the process.
//!     let gadget1 = Gadget {
//!         id: 1,
//!         owner: gadget_owner.clone(),
//!     };
//!     let gadget2 = Gadget {
//!         id: 2,
//!         owner: gadget_owner.clone(),
//!     };
//!
//!     // Dispose of our local variable `gadget_owner`.
//!     drop(gadget_owner);
//!
//!     // Despite dropping `gadget_owner`, we're still able to print out the name
//!     // of the `Owner` of the `Gadget`s. This is because we've only dropped a
//!     // single `Rc<Owner>`, not the `Owner` it points to. As long as there are
//!     // other `Rc<Owner>` values pointing at the same `Owner`, it will remain
//!     // allocated. The field projection `gadget1.owner.name` works because
//!     // `Rc<Owner>` automatically dereferences to `Owner`.
//!     println!("Gadget {} owned by {}", gadget1.id, gadget1.owner.name);
//!     println!("Gadget {} owned by {}", gadget2.id, gadget2.owner.name);
//!
//!     // At the end of the function, `gadget1` and `gadget2` are destroyed, and
//!     // with them the last counted references to our `Owner`. Gadget Man now
//!     // gets destroyed as well.
//! }
//! ```

#![feature(shared)]
#![feature(process_abort)]
#![feature(optin_builtin_traits)]
#![feature(unsize)]
#![feature(coerce_unsized)]

#[macro_use]
extern crate field_offset;

use std::borrow;
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::process::abort;
use std::marker::{self, Unsize};
use std::mem::{self, forget, size_of, size_of_val};
use std::ops::{Deref, CoerceUnsized};
use std::ptr::{self, Shared};
use std::convert::From;

#[inline(always)]
unsafe fn allocate<T>(count: usize) -> *mut T {
    let mut v = Vec::with_capacity(count);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[inline(always)]
unsafe fn deallocate<T>(ptr: *mut T, count: usize) {
    std::mem::drop(Vec::from_raw_parts(ptr, 0, count));
}

#[repr(C)]
struct RcBox<T: ?Sized> {
    strong: Cell<usize>,
    value:  T,
}

/// A single-threaded reference-counting pointer.
///
/// See the [module-level documentation](./) for more details.
///
/// The inherent methods of `Rc` are all associated functions, which means
/// that you have to call them as e.g. `Rc::get_mut(&value)` instead of
/// `value.get_mut()`.  This avoids conflicts with methods of the inner
/// type `T`.
pub struct Rc<T: ?Sized> {
    ptr: Shared<RcBox<T>>,
}

impl<T: ?Sized> !marker::Send for Rc<T> {}
impl<T: ?Sized> !marker::Sync for Rc<T> {}

impl<T, U> CoerceUnsized<Rc<U>> for Rc<T>
where T: ?Sized + Unsize<U>,
      U: ?Sized {}

impl<T> Rc<T> {
    /// Constructs a new `Rc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// ```
    pub fn new(value: T) -> Rc<T> {
        unsafe {
            let inner = RcBox {
                strong: Cell::new(1),
                value:  value,
            };

            Rc {
                ptr: Shared::new(Box::into_raw(Box::new(inner))),
            }
        }
    }

    /// Returns the contained value, if the `Rc` has exactly one strong reference.
    ///
    /// Otherwise, an [`Err`][result] is returned with the same `Rc` that was
    /// passed in.
    ///
    /// [result]: https://doc.rust-lang.org/std/result/enum.Result.html
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let x = Rc::new(3);
    /// assert_eq!(Rc::try_unwrap(x), Ok(3));
    ///
    /// let x = Rc::new(4);
    /// let _y = x.clone();
    /// assert_eq!(*Rc::try_unwrap(x).unwrap_err(), 4);
    /// ```
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        if Rc::is_unique(&this) {
            // copy the contained object
            let val = unsafe { Ok(ptr::read(&*this)) };
            forget(this);
            val
        } else {
            Err(this)
        }
    }

    /// Checks whether [`Rc::try_unwrap`][try_unwrap] would return
    /// [`Ok`].
    ///
    /// [try_unwrap]: struct.Rc.html#method.try_unwrap
    /// [`Ok`]: https://doc.rust-lang.org/std/result/enum.Result.html#variant.Ok
    pub fn would_unwrap(this: &Self) -> bool {
        Rc::is_unique(this)
    }

    /// Consumes the `Rc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Rc`
    /// using [`Rc::from_raw`][from_raw].
    ///
    /// [from_raw]: struct.Rc.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let x = Rc::new(10);
    /// let x_ptr = Rc::into_raw(x);
    /// assert_eq!(unsafe { *x_ptr }, 10);
    /// ```
    pub fn into_raw(this: Self) -> *mut T {
        let ptr = unsafe { &mut (**this.ptr).value as *mut _ };
        mem::forget(this);
        ptr
    }

    /// Constructs an `Rc` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to a
    /// [`Rc::into_raw`][into_raw].
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double-free may occur if the function is called twice on the same raw pointer.
    ///
    /// [into_raw]: struct.Rc.html#method.into_raw
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let x = Rc::new(10);
    /// let x_ptr = Rc::into_raw(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Rc` to prevent leak.
    ///     let x = Rc::from_raw(x_ptr);
    ///     assert_eq!(*x, 10);
    ///
    ///     // Further calls to `Rc::from_raw(x_ptr)` would be memory unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        // To find the corresponding pointer to the `RcBox` we need to subtract
        // the offset of the `value` field from the pointer.
        #[allow(unused_unsafe)]
        let offset = offset_of!(RcBox<T> => value).get_byte_offset();
        let rcbox = (ptr as *mut u8).offset(-(offset as isize)) as *mut _;
        Rc { ptr: Shared::new(rcbox) }
    }
}

// NOTE: We checked_add here to deal with mem::forget safety. In particular
// if you mem::forget Rcs, the ref-count can overflow, and then
// you can free the allocation while outstanding Rcs exist.
// We abort because this is such a degenerate scenario that we don't care about
// what happens -- no real program should ever experience this.
//
// This should have negligible overhead since you don't actually need to
// clone these much in Rust thanks to ownership and move-semantics.

impl<T: ?Sized> Rc<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> {
        unsafe {
            /*
            // Safe to assume this here, as if it weren't true, we'd be breaking
            // the contract anyway.
            // This allows the null check to be elided in the destructor if we
            // manipulated the reference count in the same function.
            assume(!(*(&self.ptr as *const _ as *const *const ())).is_null());
            */
            &(**self.ptr)
        }
    }

    #[inline]
    fn strong(&self) -> usize {
        self.inner().strong.get()
    }

    #[inline]
    fn inc_strong(&self) {
        self.inner().strong.set(
            self.strong().checked_add(1).unwrap_or_else(|| abort()));
    }

    #[inline]
    fn dec_strong(&self) {
        self.inner().strong.set(self.strong() - 1);
    }

    /// Gets the number of strong (`Rc`) pointers to this value.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// let _also_five = five.clone();
    ///
    /// assert_eq!(2, Rc::strong_count(&five));
    /// ```
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        this.strong()
    }

    /// Returns true if there are no other `Rc` or [`Weak`][weak] pointers to
    /// this inner value.
    ///
    /// [weak]: struct.Weak.html
    #[inline]
    pub fn is_unique(this: &Self) -> bool {
        this.strong() == 1
    }

    /// Returns a mutable reference to the inner value, if there are
    /// no other `Rc` pointers to the same value.
    ///
    /// Returns [`None`] otherwise, because it is not safe to
    /// mutate a shared value.
    ///
    /// See also [`make_mut`][make_mut], which will [`clone`][clone]
    /// the inner value when it's shared.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [make_mut]: struct.Rc.html#method.make_mut
    /// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let mut x = Rc::new(3);
    /// *Rc::get_mut(&mut x).unwrap() = 4;
    /// assert_eq!(*x, 4);
    ///
    /// let _y = x.clone();
    /// assert!(Rc::get_mut(&mut x).is_none());
    /// ```
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Rc::is_unique(this) {
            let inner = unsafe { &mut **this.ptr };
            Some(&mut inner.value)
        } else {
            None
        }
    }

    /// Returns true if the two `Rc`s point to the same value (not
    /// just values that compare as equal).
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// let same_five = five.clone();
    /// let other_five = Rc::new(5);
    ///
    /// assert!(Rc::ptr_eq(&five, &same_five));
    /// assert!(!Rc::ptr_eq(&five, &other_five));
    /// ```
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        let this_ptr:  *const RcBox<T> = *this.ptr;
        let other_ptr: *const RcBox<T> = *other.ptr;
        this_ptr == other_ptr
    }
}

impl<T: Clone> Rc<T> {
    /// Makes a mutable reference into the given `Rc`.
    ///
    /// If there are other `Rc` pointers to the same value,
    /// then `make_mut` will invoke [`clone`][clone] on the inner value to
    /// ensure unique ownership. This is also referred to as clone-on-write.
    ///
    /// See also [`get_mut`][get_mut], which will fail rather than cloning.
    ///
    /// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
    /// [get_mut]: struct.Rc.html#method.get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let mut data = Rc::new(5);
    ///
    /// *Rc::make_mut(&mut data) += 1;        // Won't clone anything
    /// let mut other_data = data.clone();    // Won't clone inner data
    /// *Rc::make_mut(&mut data) += 1;        // Clones inner data
    /// *Rc::make_mut(&mut data) += 1;        // Won't clone anything
    /// *Rc::make_mut(&mut other_data) *= 2;  // Won't clone anything
    ///
    /// // Now `data` and `other_data` point to different values.
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
    /// ```
    #[inline]
    pub fn make_mut(this: &mut Self) -> &mut T {
        if !Rc::is_unique(this) {
            // Gotta clone the data, there are other Rcs
            *this = Rc::new((**this).clone())
        }

        // This unsafety is ok because we're guaranteed that the pointer
        // returned is the *only* pointer that will ever be returned to T. Our
        // reference count is guaranteed to be 1 at this point, and we required
        // the `Rc<T>` itself to be `mut`, so we're returning the only possible
        // reference to the inner value.
        let inner = unsafe { &mut **this.ptr };
        &mut inner.value
    }
}

impl<T: ?Sized> Deref for Rc<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner().value
    }
}

impl<T: ?Sized> borrow::Borrow<T> for Rc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> AsRef<T> for Rc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> Drop for Rc<T> {
    /// Drops the `Rc`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero then we `drop` the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo  = Rc::new(Foo);
    /// let foo2 = foo.clone();
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    fn drop(&mut self) {
        unsafe {
            let ptr: *mut RcBox<T> = *self.ptr;
            self.dec_strong();
            if self.strong() == 0 {
                // run destructor of value.
                ptr::drop_in_place(&mut (*ptr).value);
                // deallocate the heap allocated memory.
                deallocate(ptr as *mut u8, size_of_val(&*ptr))
            }
        }
    }
}

impl<T: ?Sized> Clone for Rc<T> {
    /// Makes a clone of the `Rc` pointer.
    ///
    /// This creates another pointer to the same inner value, increasing the
    /// strong reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five.clone();
    /// ```
    #[inline]
    fn clone(&self) -> Rc<T> {
        self.inc_strong();
        Rc { ptr: self.ptr }
    }
}

impl<T: Default> Default for Rc<T> {
    /// Creates a new `Rc<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let x: Rc<i32> = Default::default();
    /// assert_eq!(*x, 0);
    /// ```
    #[inline]
    fn default() -> Rc<T> {
        Rc::new(Default::default())
    }
}

impl<T> From<T> for Rc<T> {
    fn from(t: T) -> Self {
        Rc::new(t)
    }
}

impl<'a, T> From<&'a [T]> for Rc<[T]> {
    /// Constructs a new `Rc<[T]>` from a shared slice [`&[T]`][slice].
    /// All elements in the slice are copied and the length is exactly that of
    /// the given [slice].
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let arr = [1, 2, 3];
    /// let rc  = Rc::from(arr);
    /// assert_eq!(rc.as_ref(), &arr);   // The elements match.
    /// assert_eq!(rc.len(), arr.len()); // The length is the same.
    /// ```
    ///
    /// Using the [`Into`][Into] trait:
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let arr          = [1, 2, 3];
    /// let rc: Rc<[u8]> = arr.as_ref().into();
    /// assert_eq!(rc.as_ref(), &arr);   // The elements match.
    /// assert_eq!(rc.len(), arr.len()); // The length is the same.
    /// ```
    ///
    /// [Into]: https://doc.rust-lang.org/std/convert/trait.Into.html
    /// [slice]: https://doc.rust-lang.org/std/primitive.slice.html
    #[inline]
    fn from(slice: &'a [T]) -> Self {
        // Compute space to allocate for `RcBox<[T]>`.
        let vptr   = slice.as_ptr();
        let vlen   = slice.len();
        let susize = size_of::<usize>();
        let aligned_len = 1 + (size_of_val(slice) + susize - 1) / susize;

        unsafe {
            // Allocate enough space for `RcBox<[T]>`.
            let ptr = allocate::<usize>(aligned_len);

            // Initialize fields of `RcBox<[T]>`.
            ptr::write(ptr, 1); // strong: Cell::new(1)
            ptr::copy_nonoverlapping(vptr, ptr.offset(1) as *mut T, vlen);

            // Combine the allocation address and the string length into a
            // fat pointer to `RcBox`.
            let rcbox_ptr: *mut RcBox<[T]> =
                mem::transmute([ptr as usize, vlen]);
            debug_assert_eq!(aligned_len * susize, size_of_val(&*rcbox_ptr));
            Rc { ptr: Shared::new(rcbox_ptr) }
        }
    }
}

impl<'a> From<&'a str> for Rc<str> {
    /// Constructs a new `Rc<str>` from a [string slice].
    /// The underlying bytes are copied from it.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let slice = "hello world!";
    /// let rc: Rc<str> = Rc::from(slice);
    /// assert_eq!(rc.as_ref(), slice);    // The elements match.
    /// assert_eq!(rc.len(), slice.len()); // The length is the same.
    /// ```
    ///
    /// Using the [`Into`][Into] trait:
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let slice = "hello world!";
    /// let rc: Rc<str> = slice.into();
    /// assert_eq!(rc.as_ref(), slice);    // The elements match.
    /// assert_eq!(rc.len(), slice.len()); // The length is the same.
    /// ```
    ///
    /// [Into]: https://doc.rust-lang.org/std/convert/trait.Into.html
    /// [string slice]: https://doc.rust-lang.org/std/primitive.str.html
    fn from(value: &'a str) -> Self {
        unsafe { mem::transmute(Rc::<[u8]>::from(value.as_bytes())) }
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Rc<T> {
    /// Equality for two `Rc`s.
    ///
    /// Two `Rc`s are equal if their inner values are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five == Rc::new(5));
    /// ```
    #[inline(always)]
    fn eq(&self, other: &Rc<T>) -> bool {
        **self == **other
    }

    /// Inequality for two `Rc`s.
    ///
    /// Two `Rc`s are unequal if their inner values are unequal.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five != Rc::new(6));
    /// ```
    #[inline(always)]
    fn ne(&self, other: &Rc<T>) -> bool {
        **self != **other
    }
}

impl<T: ?Sized + Eq> Eq for Rc<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Rc<T> {
    /// Partial comparison for two `Rc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&Rc::new(6)));
    /// ```
    #[inline(always)]
    fn partial_cmp(&self, other: &Rc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Rc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five < Rc::new(6));
    /// ```
    #[inline(always)]
    fn lt(&self, other: &Rc<T>) -> bool {
        **self < **other
    }

    /// 'Less than or equal to' comparison for two `Rc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five <= Rc::new(5));
    /// ```
    #[inline(always)]
    fn le(&self, other: &Rc<T>) -> bool {
        **self <= **other
    }

    /// Greater-than comparison for two `Rc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five > Rc::new(4));
    /// ```
    #[inline(always)]
    fn gt(&self, other: &Rc<T>) -> bool {
        **self > **other
    }

    /// 'Greater than or equal to' comparison for two `Rc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(five >= Rc::new(5));
    /// ```
    #[inline(always)]
    fn ge(&self, other: &Rc<T>) -> bool {
        **self >= **other
    }
}

impl<T: ?Sized + Ord> Ord for Rc<T> {
    /// Comparison for two `Rc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&Rc::new(6)));
    /// ```
    #[inline]
    fn cmp(&self, other: &Rc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Hash> Hash for Rc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&*self.ptr, f)
    }
}