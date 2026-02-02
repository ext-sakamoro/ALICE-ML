//! Arena Allocator for Zero-Allocation Inference
//!
//! Pre-allocates all memory upfront. No runtime allocations during inference.

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
        let size = core::mem::size_of::<T>() * count;

        // Align offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + size > self.buffer.len() {
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
        assert_eq!(floats.len(), 10);

        let ints = arena.alloc::<i32>(20).unwrap();
        assert_eq!(ints.len(), 20);

        assert!(arena.used() > 0);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new(1024);

        let _ = arena.alloc::<f32>(100).unwrap();
        let used_before = arena.used();

        arena.reset();
        assert_eq!(arena.used(), 0);

        let _ = arena.alloc::<f32>(100).unwrap();
        assert_eq!(arena.used(), used_before);
    }

    #[test]
    fn test_arena_exhaustion() {
        let mut arena = Arena::new(64);

        // Should succeed
        let _ = arena.alloc::<f32>(10).unwrap();

        // Should fail (not enough space)
        assert!(arena.alloc::<f32>(100).is_none());
    }

    #[test]
    fn test_arena_alignment() {
        let mut arena = Arena::new(1024);

        // Allocate unaligned first
        let _ = arena.alloc::<u8>(3).unwrap();

        // Next allocation should still be aligned
        let floats = arena.alloc::<f32>(4).unwrap();
        let ptr = floats.as_ptr() as usize;
        assert_eq!(ptr % core::mem::align_of::<f32>(), 0);
    }
}
