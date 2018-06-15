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

#![feature(unsize, coerce_unsized)]
#![feature(specialization)]
#![feature(allocator_api)]

use std::alloc::{self, Layout};
use std::borrow;
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::process::abort;
use std::marker::{PhantomData, Unsize};
use std::mem::{self, forget, size_of_val, align_of_val};
use std::ops::{Deref, CoerceUnsized};
use std::ptr::{self, NonNull};

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
    ptr: NonNull<RcBox<T>>,
    phantom: PhantomData<T>,
}

impl<T, U> CoerceUnsized<Rc<U>> for Rc<T>
where T: ?Sized + Unsize<U>,
      U: ?Sized {}

unsafe fn set_ptr<T:?Sized>(mut fat: *const T,  alloc: *mut u8) -> *mut RcBox<T> {
    ptr::write(&mut fat as *mut *const T as *mut *const u8, alloc);
    fat as *mut RcBox<T>
}

impl<T> Rc<T> {
    /// Constructs a new `Rc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// std::mem::drop(five);
    /// ```
    pub fn new(value: T) -> Rc<T> {
        unsafe {
            let inner = RcBox {
                strong: Cell::new(1),
                value:  value,
            };

            Rc {
                ptr: NonNull::new_unchecked(Box::into_raw(Box::new(inner))),
                phantom: PhantomData,
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
        let ptr = unsafe { &this.ptr.as_ref().value as *const _ as *mut _ };
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
        // Align the unsized value to the end of the RcBox.
        // Because it is ?Sized, it will always be the last field in memory.
        let align = align_of_val(&*ptr);
        let layout = Layout::new::<RcBox<()>>();
        let offset = (layout.size() + layout.padding_needed_for(align)) as isize;

        // Reverse the offset to find the original RcBox.
        let rcbox_ptr = set_ptr(ptr as *const T, (ptr as *mut u8).offset(-offset));

        Rc {
            ptr: NonNull::new_unchecked(rcbox_ptr),
            phantom: PhantomData,
        }
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
            self.ptr.as_ref()
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
            let inner = unsafe { this.ptr.as_mut() };
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
        this.ptr == other.ptr
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
        unsafe { &mut this.ptr.as_mut().value }
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

#[inline(never)]
unsafe fn slow_drop<T:?Sized>(ptr: *mut RcBox<T>) {
    let layout = Layout::for_value(&*ptr);
    // run destructor of value.
    ptr::drop_in_place(&mut(*ptr).value);
    // deallocate the heap allocated memory.
    alloc::dealloc(ptr as *mut u8, layout);
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
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.dec_strong();
            if self.strong() == 0 {
                slow_drop(self.ptr.as_ptr());
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
        Rc { ptr: self.ptr, phantom: PhantomData }
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

impl<T: ?Sized> From<Box<T>> for Rc<T> {
    /// Constructs a new `Rc<T>` from a `Box<T>` where `T` can be unsized.
    /// The allocated space of the `Box<T>` is not reused,
    /// but new space is allocated and the old is deallocated.
    /// This happens due to the internal layout of `Rc`.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let arr = [1, 2, 3];
    /// let vec = vec![Box::new(1), Box::new(2), Box::new(3)].into_boxed_slice();
    /// let rc: Rc<[Box<usize>]> = Rc::from(vec);
    /// assert_eq!(rc.len(), arr.len());
    /// for (x, y) in rc.iter().zip(&arr) {
    ///     assert_eq!(**x, *y);
    /// }
    /// ```
    #[inline]
    fn from(boxed: Box<T>) -> Self {
        unsafe {
            // Compute space to allocate + alignment for `RcBox<T>`.
            let fake_rcbox_ptr = &*boxed as *const T as *const RcBox<T>;
            let rcbox_layout = Layout::for_value(&*fake_rcbox_ptr);

            // Allocate the space.
            let alloc  = alloc::alloc(rcbox_layout);

            // Cast to fat pointer: *mut RcBox<T>.
            let bptr      = Box::into_raw(boxed);
            let rcbox_ptr = set_ptr(bptr as *const T, alloc);

            // Initialize fields of RcBox<T>.
            (*rcbox_ptr).strong.set(1);
            let box_layout = Layout::for_value(&*bptr);
            ptr::copy_nonoverlapping(
                bptr as *const u8,
                (&mut (*rcbox_ptr).value) as *mut T as *mut u8,
                box_layout.size());

            // Deallocate box, Box::into_raw consumed it and we've moved the value
            alloc::dealloc(bptr as *mut u8, box_layout);

            // Yield the Rc:
            debug_assert_eq!(rcbox_layout.size(), mem::size_of_val(&*rcbox_ptr));
            Rc {
                ptr: NonNull::new_unchecked(rcbox_ptr),
                phantom: PhantomData,
            }
        }
    }
}

#[inline(always)]
unsafe fn slice_to_rc<'a, T, U, W, C>(src: &'a [T], cast: C, write_elems: W)
   -> Rc<U>
where U: ?Sized,
      W: FnOnce(&mut [T], &[T]),
      C: FnOnce(*mut RcBox<[T]>) -> *mut RcBox<U> {
    // Compute space to allocate for `RcBox<U>` by creating a fake pointer.
    // (this is how std does it)
    let fake_rcbox_ptr = src as *const [T] as *const RcBox<[T]>;
    let layout = Layout::for_value(&*fake_rcbox_ptr);

    // Allocate the space.
    let alloc: *mut u8 = alloc::alloc(layout);

    // Combine the allocation address and the slice length into a
    // fat pointer to RcBox<[T]>.
    let rcbox_ptr = set_ptr(src as *const [T], alloc);

    // Initialize fields of RcBox<[T]>.
    (*rcbox_ptr).strong.set(1);
    write_elems(&mut (*rcbox_ptr).value, src);

    // cast to the final type (either [T] or str)
    let rcbox_ptr = cast(rcbox_ptr);
    debug_assert_eq!(layout.size(), size_of_val(&*rcbox_ptr));
    Rc {
        ptr: NonNull::new_unchecked(rcbox_ptr),
        phantom: PhantomData
    }
}

impl<T> From<Vec<T>> for Rc<[T]> {
    /// Constructs a new `Rc<[T]>` from a `Vec<T>`.
    /// The allocated space of the `Vec<T>` is not reused,
    /// but new space is allocated and the old is deallocated.
    /// This happens due to the internal layout of `Rc`.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// let arr = [1, 2, 3];
    /// let vec = vec![Box::new(1), Box::new(2), Box::new(3)];
    /// let rc: Rc<[Box<usize>]> = Rc::from(vec);
    /// assert_eq!(rc.len(), arr.len());
    /// for (x, y) in rc.iter().zip(&arr) {
    ///     assert_eq!(**x, *y);
    /// }
    /// ```
    #[inline]
    fn from(mut vec: Vec<T>) -> Self {
        unsafe {
            let rc = slice_to_rc(vec.as_slice(), |p| p, |dst, src|
                ptr::copy_nonoverlapping(
                    src.as_ptr(), dst.as_mut_ptr(), src.len())
            );
            // Prevent vec from trying to drop the elements:
            vec.set_len(0);
            rc
        }
    }
}

impl<'a, T: Clone> From<&'a [T]> for Rc<[T]> {
    /// Constructs a new `Rc<[T]>` by cloning all elements from the shared slice
    /// [`&[T]`][slice]. The length of the reference counted slice will be exactly
    /// the given [slice].
    ///
    /// If a call to `clone()` unwinds, already cloned elements and the memory
    /// for the `Rc` might not be dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// #[derive(PartialEq, Clone, Debug)]
    /// struct Wrap(u8);
    ///
    /// let arr = [Wrap(1), Wrap(2), Wrap(3)];
    /// let rc: Rc<[Wrap]> = Rc::from(arr.as_ref());
    /// assert_eq!(rc.as_ref(), &arr);   // The elements match.
    /// assert_eq!(rc.len(), arr.len()); // The lengths match.
    /// ```
    ///
    /// Using the [`Into`][Into] trait:
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// #[derive(PartialEq, Clone, Debug)]
    /// struct Wrap(u8);
    ///
    /// let arr = [Wrap(1), Wrap(2), Wrap(3)];
    /// let rc: Rc<[Wrap]> = arr.as_ref().into();
    /// assert_eq!(rc.as_ref(), &arr);   // The elements match.
    /// assert_eq!(rc.len(), arr.len()); // The lengths match.
    /// ```
    ///
    /// [Into]: https://doc.rust-lang.org/std/convert/trait.Into.html
    /// [slice]: https://doc.rust-lang.org/std/primitive.slice.html
    #[inline]
    default fn from(slice: &'a [T]) -> Self {
        unsafe {
            slice_to_rc(slice, |p| p, |dst, src| {
                for (d, s) in dst.iter_mut().zip(src) {
                    ptr::write(d, s.clone())
                }
            })
        }
    }
}

impl<'a, T: Copy> From<&'a [T]> for Rc<[T]> {
    /// Constructs a new `Rc<[T]>` from a shared slice [`&[T]`][slice].
    /// All elements in the slice are copied and the length is exactly that of
    /// the given [slice]. In this case, `T` must be `Copy`.
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
    /// [Into]: ../../std/convert/trait.Into.html
    /// [slice]: ../../std/primitive.slice.html
    #[inline]
    fn from(slice: &'a [T]) -> Self {
        unsafe {
            slice_to_rc(slice, |p| p, <[T]>::copy_from_slice)
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
    /// This can be useful in doing [string interning], and cache:ing your strings.
    ///
    /// ```
    /// use strong_rc::Rc;
    ///
    /// use std::collections::HashSet;
    /// use std::mem::drop;
    ///
    /// fn cache_str(cache: &mut HashSet<Rc<str>>, input: &str) -> Rc<str> {
    ///     // If the input hasn't been cached, do it:
    ///     if !cache.contains(input) {
    ///         cache.insert(input.into());
    ///     }
    ///
    ///     // Retrieve the cached element.
    ///     cache.get(input).unwrap().clone()
    /// }
    ///
    /// let first   = "hello world!";
    /// let second  = "goodbye!";
    /// let mut set = HashSet::new();
    ///
    /// // Cache the slices:
    /// let rc_first  = cache_str(&mut set, first);
    /// let rc_second = cache_str(&mut set, second);
    /// let rc_third  = cache_str(&mut set, second);
    ///
    /// // The contents match:
    /// assert_eq!(rc_first.as_ref(),  first);
    /// assert_eq!(rc_second.as_ref(), second);
    /// assert_eq!(rc_third.as_ref(),  rc_second.as_ref());
    ///
    /// // It was cached:
    /// assert_eq!(set.len(), 2);
    /// drop(set);
    /// assert_eq!(Rc::strong_count(&rc_first),  1);
    /// assert_eq!(Rc::strong_count(&rc_second), 2);
    /// assert_eq!(Rc::strong_count(&rc_third),  2);
    /// assert!(Rc::ptr_eq(&rc_second, &rc_third));
    /// ```
    ///
    /// [string interning]: https://en.wikipedia.org/wiki/String_interning
    fn from(slice: &'a str) -> Self {
        // This is safe since the input was valid utf8 to begin with, and thus
        // the invariants hold.
        unsafe {
            let bytes = slice.as_bytes();
            slice_to_rc(bytes, |p| p as *mut RcBox<str>, <[u8]>::copy_from_slice)
        }
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
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

/// A hidden place for negative trait tests.
///
/// !Sync:
/// ```compile_fail
/// use strong_rc::Rc;
/// fn sync<T:Sync>(_: T) {}
/// sync(Rc::new(()));
/// ```
///
/// !Send:
/// ```compile_fail
/// use strong_rc::Rc;
/// fn send<T:Send>(_: T) {}
/// send(Rc::new(()));
/// ```
#[allow(unused)]
struct ThreadsafeTest;
