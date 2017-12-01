#![deny(warnings)]

extern crate either;

#[cfg(test)]
extern crate crossbeam;

extern crate crossbeam_epoch as epoch;
extern crate lazy_init;


use either::Either;

use lazy_init::LazyTransform as LazyTransformSync;
use epoch::{Atomic, Owned};
use std::sync::atomic::Ordering;


// Note:
// 1. Transformation function is passed in `get` method,
//    rather than in constructor (`new`). (rust-lang/rust/issues/44490)
// 2. Thread-safe `get` returns `Option<B>` (by value, not by reference).
//    And hence `LazyTransformStoreSync<A, B>` requires `B: Clone`.
// 3. `LazyTransformStoreSync<A, B>` allocates
//    and uses epoch-based memory reclamation.


#[derive(Debug, Clone)]
pub struct LazyTransformStore<A, B> {
    value: Option<Either<A, B>>,
}

impl<A, B> Default for LazyTransformStore<A, B> {
    fn default() -> Self {
        Self { value: None }
    }
}

impl<A, B> LazyTransformStore<A, B> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn set(&mut self, value: A) {
        self.value = Some(Either::Left(value));
    }

    pub fn get<F>(&mut self, f: F) -> Option<&B>
    where
        F: FnOnce(A) -> B,
    {
        self.transform_if_needed(f);
        self.value
            .as_ref()
            .map(Either::as_ref)
            .and_then(Either::right)
    }

    fn transform_if_needed<F>(&mut self, f: F)
    where
        F: FnOnce(A) -> B,
    {
        let transformed = self.value
            .take()
            .map(|e| e.left_and_then(|v| Either::Right(f(v))));
        self.value = transformed;
    }
}


/// Same as the `LazyTransformStore`, but thread-safe.
#[derive(Debug)]
pub struct LazyTransformStoreSync<A, B>
where
    A: Sync,
    B: Sync + Clone,
{
    ptr: Atomic<LazyTransformSync<A, B>>,
}

impl<A, B> Default for LazyTransformStoreSync<A, B>
where
    A: Sync,
    B: Sync + Clone,
{
    fn default() -> Self {
        Self {
            ptr: Atomic::null(),
        }
    }
}

unsafe impl<A, B> Sync for LazyTransformStoreSync<A, B>
where
    A: Sync,
    B: Sync + Clone,
{
}

impl<A, B> LazyTransformStoreSync<A, B>
where
    A: Sync,
    B: Sync + Clone,
{
    pub fn new() -> Self {
        Default::default()
    }

    pub fn set(&self, value: A) {
        // By pinning a participant (thread) we declare, that any object,
        // that gets removed from now on, must not be destructed just yet.
        // Garbage collection of newly removed objects is suspended
        // until the participant gets unpinned.
        let guard = &epoch::pin();
        let old = self.ptr.swap(
            // Allocate a new lazy init-once object.
            Owned::new(LazyTransformSync::new(value)).into_shared(guard),
            Ordering::AcqRel,
            guard,
        );
        // The object `old` is pointing to is now unreachable.
        // Defer its deallocation until all currently pinned threads get unpinned.
        if !old.is_null() {
            unsafe {
                guard.defer(move || drop(old.into_owned()));
            }
        }
    }

    pub fn get<F>(&self, f: F) -> Option<B>
    where
        F: FnOnce(A) -> B,
    {
        let guard = &epoch::pin();
        let snapshot = self.ptr.load(Ordering::Acquire, guard);

        if snapshot.is_null() {
            return None;
        }

        let lazy = unsafe { snapshot.deref() };
        // blocking call
        Some(lazy.get_or_create(f).clone())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn lazy_transform_store() {
        // to ensure laziness
        let counter = Cell::new(0);
        let mut lazy = LazyTransformStore::new();
        assert_eq!(lazy.get(|_| 42), None);
        lazy.set(3);
        assert_eq!(counter.get(), 0);
        assert_eq!(
            lazy.get(|x| {
                counter.set(counter.get() + 1);
                x * 2
            }),
            Some(&6)
        );
        assert_eq!(counter.get(), 1);
        lazy.set(10);
        assert_eq!(counter.get(), 1);
        assert_eq!(
            lazy.get(|x| {
                counter.set(counter.get() + 1);
                x * 2
            }),
            Some(&20)
        );
        assert_eq!(counter.get(), 2);
        assert_eq!(lazy.get(|_| 42), Some(&20));
        assert_eq!(counter.get(), 2);
    }

    #[test]
    fn stress_lazy_transform_store_sync() {
        let lazy = LazyTransformStoreSync::new();

        assert_eq!(lazy.get(|v| v), None);
        lazy.set(1);
        assert_eq!(lazy.get(|v| v), Some(1));
        assert_eq!(lazy.get(|v| v), Some(1));
        lazy.set(2);
        assert_eq!(lazy.get(|v| v), Some(2));

        crossbeam::scope(|scope| {
            for _ in 0..16 {
                scope.spawn(|| {
                    for i in 0..128 {
                        // call `set` 1/4 of the time
                        if i % 4 == 0 {
                            thread::sleep(Duration::from_millis(5));
                            lazy.set(i);
                        } else {
                            lazy.get(|v| {
                                thread::sleep(Duration::from_millis(15));
                                v * v
                            });
                            assert!(lazy.get(|v| v).is_some());
                        }
                    }
                });
            }
        });
    }
}
