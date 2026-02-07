//! Arena Allocator for Zero-Allocation Inference
//!
//! Pre-allocates all memory upfront. No runtime allocations during inference.
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Simple bump allocator for inference buffers
///
/// All intermediate tensors are allocated from this arena.
/// Reset between inference calls for zero-allocation operation.
pub struct Arena {
    buffer: Vec<u8>,
    offset: usize,
}

impl Arena {
    /// Create arena with specified capacity (bytes)
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            offset: 0,
        }
    }

    /// Allocate aligned memory from arena
    ///
    /// Returns None if arena is exhausted.
    #[inline]
    pub fn alloc<T>(&mut self, count: usize) -> Option<&mut [T]> {
        let align = core::mem::align_of::<T>();
        let size = core::mem::size_of::<T>().checked_mul(count)?;

        // Align offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset.checked_add(size)? > self.buffer.len() {
            return None;
        }

        let ptr = unsafe {
            self.buffer.as_mut_ptr().add(aligned_offset) as *mut T
        };

        self.offset = aligned_offset + size;

        Some(unsafe { core::slice::from_raw_parts_mut(ptr, count) })
    }

    /// Allocate and zero-initialize
    #[inline]
    pub fn alloc_zeroed<T: Default + Clone>(&mut self, count: usize) -> Option<&mut [T]> {
        let slice = self.alloc::<T>(count)?;
        for item in slice.iter_mut() {
            *item = T::default();
        }
        Some(slice)
    }

    /// Reset arena for reuse (no deallocation, just reset offset)
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Current usage in bytes
    #[inline]
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Total capacity in bytes
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Remaining capacity in bytes
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.offset
    }
}

impl Default for Arena {
    fn default() -> Self {
        // Default 16MB arena
        Self::new(16 * 1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc() {
        let mut arena = Arena::new(1024);

        let floats = arena.alloc::<f32>(10).unwrap();
        assert_eq!(floats.len(), 10, "should allocate 10 floats");

        let ints = arena.alloc::<i32>(20).unwrap();
        assert_eq!(ints.len(), 20, "should allocate 20 ints");

        assert!(arena.used() > 0, "arena should track usage");
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new(1024);

        let _ = arena.alloc::<f32>(100).unwrap();
        let used_before = arena.used();

        arena.reset();
        assert_eq!(arena.used(), 0, "reset should clear usage");

        let _ = arena.alloc::<f32>(100).unwrap();
        assert_eq!(arena.used(), used_before, "re-alloc should use same amount");
    }

    #[test]
    fn test_arena_exhaustion() {
        let mut arena = Arena::new(64);

        // Should succeed
        let _ = arena.alloc::<f32>(10).unwrap();

        // Should fail (not enough space)
        assert!(arena.alloc::<f32>(100).is_none(), "should return None when exhausted");
    }

    #[test]
    fn test_arena_alignment() {
        let mut arena = Arena::new(1024);

        // Allocate unaligned first
        let _ = arena.alloc::<u8>(3).unwrap();

        // Next allocation should still be aligned
        let floats = arena.alloc::<f32>(4).unwrap();
        let ptr = floats.as_ptr() as usize;
        assert_eq!(ptr % core::mem::align_of::<f32>(), 0, "f32 allocation must be aligned");
    }

    #[test]
    fn test_arena_overflow_protection() {
        let mut arena = Arena::new(1024);
        // Requesting usize::MAX elements should return None, not panic
        assert!(arena.alloc::<f32>(usize::MAX).is_none(), "overflow should return None");
    }
}
