#![feature(test)]
extern crate test;
extern crate strong_rc;

use strong_rc::Rc;
use test::Bencher;
use std::mem;

#[bench]
fn std_clone(b: &mut Bencher) {
    use std::rc::Rc;
    let rc = Rc::new(1);
    // hopefully bencher needs less than 2^31 iterations...
    b.iter(|| mem::forget(rc.clone()));
}

#[bench]
fn rc_clone(b: &mut Bencher) {
    let rc = Rc::new(1);
    // hopefully bencher needs less than 2^31 iterations...
    b.iter(|| mem::forget(rc.clone()));
}

#[bench]
fn std_drop(b: &mut Bencher) {
    use std::rc::Rc;
    let rc = Rc::new(1);
    b.iter(|| mem::drop(rc.clone()));
}

#[bench]
fn rc_drop(b: &mut Bencher) {
    let rc = Rc::new(1);
    b.iter(|| mem::drop(rc.clone()));
}

#[bench]
fn std_str(b: &mut Bencher) {
    use std::rc::Rc;
    b.iter(|| Rc::<str>::from("hello world"));
}

#[bench]
fn rc_str(b: &mut Bencher) {
    b.iter(|| Rc::<str>::from("hello world"));
}

#[bench]
fn std_slice(b: &mut Bencher) {
    use std::rc::Rc;
    b.iter(|| Rc::<[u8]>::from([].as_ref()));
}

#[bench]
fn rc_slice(b: &mut Bencher) {
    b.iter(|| Rc::<[u8]>::from([].as_ref()));
}
