extern crate rand;

extern crate strong_rc;

use strong_rc::Rc;

use std::boxed::Box;
use std::mem::drop;
use std::fmt;

#[test]
fn clone() {
    use std::cell::RefCell;
    let x = Rc::new(RefCell::new(5));
    let y = x.clone();
    *x.borrow_mut() = 20;
    assert_eq!(*y.borrow(), 20);
}

#[test]
fn simple() {
    let x = Rc::new(5);
    assert_eq!(*x, 5);
}

#[test]
fn simple_clone() {
    let x = Rc::new(5);
    let y = x.clone();
    assert_eq!(*x, 5);
    assert_eq!(*y, 5);
}

#[test]
fn destructor() {
    let x: Rc<Box<_>> = Rc::new(Box::new(5));
    assert_eq!(**x, 5);
}

#[test]
fn is_unique() {
    let x = Rc::new(3);
    assert!(Rc::is_unique(&x));
    let ys: Vec<_> = (0..10).map(|_| {
        let y = x.clone();
        assert!(!Rc::is_unique(&x));
        y
    }).collect();
    drop(ys);
    assert!(Rc::is_unique(&x));
}

#[test]
fn strong_count() {
    let x = Rc::new(0);
    assert_eq!(Rc::strong_count(&x), 1);
    let ys: Vec<_> = (2..10).map(|c| {
        let y = x.clone();
        assert_eq!(Rc::strong_count(&y), c);
        y
    }).collect();
    drop(ys);
    assert_eq!(Rc::strong_count(&x), 1);
}

#[test]
fn try_unwrap() {
    let x = Rc::new(3);
    assert_eq!(Rc::try_unwrap(x), Ok(3));
    let x = Rc::new(4);
    let _y = x.clone();
    assert_eq!(Rc::try_unwrap(x), Err(Rc::new(4)));
}

#[test]
fn try_unwrap_box() {
    let x = Rc::new(Box::new("hello"));

    assert_eq!(**x, "hello");
    let y = x.clone();
    assert_eq!(Rc::strong_count(&x), 2);
    drop(y);

    let uw = Rc::try_unwrap(x);
    assert_eq!(uw.map(|x| *x), Ok("hello"));
}

#[test]
fn get_mut() {
    let mut x = Rc::new(3);
    *Rc::get_mut(&mut x).unwrap() = 4;
    assert_eq!(*x, 4);
    let y = x.clone();
    assert!(Rc::get_mut(&mut x).is_none());
    drop(y);
    assert!(Rc::get_mut(&mut x).is_some());
}

#[test]
fn into_from_raw() {
    let x = Rc::new(Box::new("hello"));
    let y = x.clone();
    let x_ptr = Rc::into_raw(x);

    drop(y);

    unsafe {
        assert_eq!(**x_ptr, "hello");
        let x = Rc::from_raw(x_ptr);
        assert_eq!(**x, "hello");

        let y = x.clone();
        assert_eq!(Rc::strong_count(&x), 2);
        drop(y);

        let uw = Rc::try_unwrap(x);
        assert_eq!(uw.map(|x| *x), Ok("hello"));
    }
}

#[test]
fn into_from_raw2() {
    let x = Rc::<usize>::new(1);
    let y = x.clone();

    let x_ptr = Rc::into_raw(x);

    drop(y);

    unsafe {
        assert_eq!(*x_ptr, 1);
        let x = Rc::<usize>::from_raw(x_ptr);
        assert_eq!(*x, 1);
        assert_eq!(Rc::try_unwrap(x), Ok(1));
    }
}

#[test]
fn cowrc_clone_make_unique() {
    let mut cow0 = Rc::new(75);
    let mut cow1 = cow0.clone();
    let mut cow2 = cow1.clone();

    assert_eq!(75, *Rc::make_mut(&mut cow0));
    assert_eq!(75, *Rc::make_mut(&mut cow1));
    assert_eq!(75, *Rc::make_mut(&mut cow2));

    *Rc::make_mut(&mut cow0) += 1;
    *Rc::make_mut(&mut cow1) += 2;
    *Rc::make_mut(&mut cow2) += 3;

    assert_eq!(76, *cow0);
    assert_eq!(77, *cow1);
    assert_eq!(78, *cow2);

    // none should point to the same backing memory
    assert_ne!(*cow0, *cow1);
    assert_ne!(*cow0, *cow2);
    assert_ne!(*cow1, *cow2);
}

#[test]
fn cowrc_clone_unique2() {
    let mut cow0 = Rc::new(75);
    let cow1 = cow0.clone();
    let cow2 = cow1.clone();

    assert_eq!(75, *cow0);
    assert_eq!(75, *cow1);
    assert_eq!(75, *cow2);

    *Rc::make_mut(&mut cow0) += 1;

    assert_eq!(76, *cow0);
    assert_eq!(75, *cow1);
    assert_eq!(75, *cow2);

    // cow1 and cow2 should share the same contents
    // cow0 should have a unique reference
    assert_ne!(*cow0, *cow1);
    assert_ne!(*cow0, *cow2);
    assert_eq!(*cow1, *cow2);
}

#[test]
fn show() {
    let foo = Rc::new(75);
    assert_eq!(format!("{:?}", foo), "75");
}

#[test]
fn rc_unsized() {
    let foo: Rc<[i32]> = Rc::new([1, 2, 3]);
    assert_eq!(foo, foo.clone());
}

#[test]
fn str() {
    let x: Rc<str> = "hello".into();
    let y = x.clone();
    assert_eq!(Rc::strong_count(&x), 2);
    assert_eq!(x.len(), "hello".len());
    assert_eq!(y.len(), "hello".len());
    drop(y);
    assert_eq!(x.as_ref(), "hello");
    assert_eq!(Rc::strong_count(&x), 1);
}

#[test]
fn from_owned() {
    let foo = 123;
    let foo_rc = Rc::from(foo);
    assert_eq!(123, *foo_rc);
}

#[test]
fn ptr_eq() {
    let five = Rc::new(5);
    let same_five = five.clone();
    let other_five = Rc::new(5);

    assert!(Rc::ptr_eq(&five, &same_five));
    assert!(!Rc::ptr_eq(&five, &other_five));
    drop(five);
}

#[test]
fn rc_slice() {
    fn make_rvec<T: Clone + rand::Rand>(n: usize) -> Vec<T> {
        use rand::Rng;
        let mut vec: Vec<T> = Vec::with_capacity(n);
        let mut rng = rand::thread_rng();
        for _ in 0 .. n {
            vec.push(rng.gen());
        }
        vec
    }

    fn check_all<T: PartialEq + fmt::Debug>(rc: Rc<[T]>, vec: Vec<T>, n: usize) {
        assert_eq!(rc.len(), n);
        for (x, v) in rc.iter().zip(vec.into_iter()) {
            assert_eq!(*x, v);
        }
    }

    fn test<T: Copy + PartialEq + fmt::Debug + rand::Rand>(n: usize) {
        let vec = make_rvec(n);
        check_all(Rc::<[T]>::from(vec.as_ref()), vec, n);
    }

    fn testcl<T: Clone + PartialEq + fmt::Debug + rand::Rand>(n: usize) {
        let vec = make_rvec(n);
        check_all(Rc::<[T]>::from(vec.as_ref()), vec, n);
    }

    #[repr(packed)]
    #[derive(PartialEq, Clone, Copy, Debug)]
    struct Pack(u8, u8, u8);

    #[derive(PartialEq, Clone, Debug)]
    struct Cl(u8);

    impl rand::Rand for Pack {
        fn rand<R: rand::Rng>(rng: &mut R) -> Self {
            Pack(u8::rand(rng), u8::rand(rng), u8::rand(rng))
        }
    }

    impl rand::Rand for Cl {
        fn rand<R: rand::Rng>(rng: &mut R) -> Self {
            Cl(u8::rand(rng))
        }
    }

    for n in 0..100 {
        test::<u8>(n);
        test::<Pack>(n);
        testcl::<Cl>(n);
        test::<u16>(n);
        test::<u32>(n);
        test::<u64>(n);
        test::<usize>(n);
    }
}