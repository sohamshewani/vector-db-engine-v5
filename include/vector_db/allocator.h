#ifndef VECTOR_DB_ALLOCATOR_H
#define VECTOR_DB_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <sys/types.h>

#define VECTOR_DB_CACHE_LINE 64
#define VECTOR_DB_ARENA_MAGIC 0x5644424152454E41ULL /* "VDBARENA" */

typedef enum {
    VDB_ALLOC_SUCCESS = 0,
    VDB_ALLOC_ERR_OOM = -1,
    VDB_ALLOC_ERR_INVALID_ARG = -2,
    VDB_ALLOC_ERR_CORRUPTION = -3,
    VDB_ALLOC_ERR_THREAD_LOCAL = -4
} vdb_alloc_status_t;

typedef struct vdb_arena {
    uint64_t magic;
    uint8_t *base_ptr;
    size_t capacity;
    _Atomic size_t offset;
    bool is_thread_local;
    int node_id;
    struct vdb_arena *next;
} vdb_arena_t;

typedef struct vdb_thread_cache {
    vdb_arena_t *local_arena;
    uint64_t allocations_count;
    uint64_t bytes_allocated;
} vdb_thread_cache_t;

vdb_alloc_status_t vdb_arena_create(vdb_arena_t **out_arena, size_t capacity, int node_id);
void vdb_arena_destroy(vdb_arena_t *arena);
vdb_alloc_status_t vdb_arena_allocate(vdb_arena_t *arena, size_t size, size_t alignment, void **out_ptr);
void vdb_arena_reset(vdb_arena_t *arena);

vdb_alloc_status_t vdb_thread_cache_init(void);
void vdb_thread_cache_cleanup(void);
vdb_alloc_status_t vdb_thread_local_allocate(size_t size, size_t alignment, void **out_ptr);

void *vdb_aligned_alloc(size_t alignment, size_t size);
void vdb_aligned_free(void *ptr);

#endif /* VECTOR_DB_ALLOCATOR_H */

#ifndef VECTOR_DB_ALLOCATOR_C
#define VECTOR_DB_ALLOCATOR_C

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
#include <unistd.h>
#endif

static _Thread_local vdb_thread_cache_t tls_cache = {
    .local_arena = NULL,
    .allocations_count = 0,
    .bytes_allocated = 0
};

static _Atomic bool tls_initialized = false;

static inline size_t vdb_align_forward(size_t addr, size_t alignment) {
    size_t modulo = addr & (alignment - 1);
    if (modulo == 0) {
        return addr;
    }
    return addr + (alignment - modulo);
}

void *vdb_aligned_alloc(size_t alignment, size_t size) {
    if (alignment < VECTOR_DB_CACHE_LINE) {
        alignment = VECTOR_DB_CACHE_LINE;
    }
    
    // Ensure size is a multiple of alignment
    size_t remainder = size % alignment;
    if (remainder != 0) {
        size += (alignment - remainder);
    }

    void *ptr = NULL;
#if defined(_ISOC11_SOURCE) || __STDC_VERSION__ >= 201112L
    ptr = aligned_alloc(alignment, size);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL;
    }
#else
    // Fallback implementation with offset tracking
    size_t total_size = size + alignment + sizeof(void*);
    void *raw = malloc(total_size);
    if (!raw) return NULL;
    
    uintptr_t raw_addr = (uintptr_t)raw;
    uintptr_t body_addr = vdb_align_forward(raw_addr + sizeof(void*), alignment);
    ptr = (void*)body_addr;
    *((void**)(body_addr - sizeof(void*))) = raw;
#endif
    return ptr;
}

void vdb_aligned_free(void *ptr) {
    if (!ptr) {
        return;
    }
#if defined(_ISOC11_SOURCE) || __STDC_VERSION__ >= 201112L || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
    free(ptr);
#else
    void *raw = *((void**)((uintptr_t)ptr - sizeof(void*)));
    free(raw);
#endif
}

vdb_alloc_status_t vdb_arena_create(vdb_arena_t **out_arena, size_t capacity, int node_id) {
    if (!out_arena) {
        return VDB_ALLOC_ERR_INVALID_ARG;
    }
    
    if (capacity == 0) {
        capacity = 1024 * 1024 * 64; // Default 64MB arena
    }

    // Align capacity to cache line
    size_t rem = capacity % VECTOR_DB_CACHE_LINE;
    if (rem != 0) {
        capacity += (VECTOR_DB_CACHE_LINE - rem);
    }

    vdb_arena_t *arena = (vdb_arena_t*)vdb_aligned_alloc(VECTOR_DB_CACHE_LINE, sizeof(vdb_arena_t));
    if (!arena) {
        return VDB_ALLOC_ERR_OOM;
    }

    uint8_t *base = (uint8_t*)vdb_aligned_alloc(VECTOR_DB_CACHE_LINE, capacity);
    if (!base) {
        vdb_aligned_free(arena);
        return VDB_ALLOC_ERR_OOM;
    }

    arena->magic = VECTOR_DB_ARENA_MAGIC;
    arena->base_ptr = base;
    arena->capacity = capacity;
    atomic_init(&arena->offset, 0);
    arena->is_thread_local = false;
    arena->node_id = node_id;
    arena->next = NULL;

    *out_arena = arena;
    return VDB_ALLOC_SUCCESS;
}

void vdb_arena_destroy(vdb_arena_t *arena) {
    if (!arena) {
        return;
    }
    if (arena->magic != VECTOR_DB_ARENA_MAGIC) {
        // Corruption detected, abort structural free
        return;
    }
    arena->magic = 0; // Poison
    vdb_aligned_free(arena->base_ptr);
    vdb_aligned_free(arena);
}

vdb_alloc_status_t vdb_arena_allocate(vdb_arena_t *arena, size_t size, size_t alignment, void **out_ptr) {
    if (!arena || !out_ptr) {
        return VDB_ALLOC_ERR_INVALID_ARG;
    }
    if (arena->magic != VECTOR_DB_ARENA_MAGIC) {
        return VDB_ALLOC_ERR_CORRUPTION;
    }
    if (alignment == 0) {
        alignment = VECTOR_DB_CACHE_LINE;
    }

    size_t current_offset = atomic_load_explicit(&arena->offset, memory_order_relaxed);
    size_t aligned_offset = 0;
    size_t new_offset = 0;

    while (true) {
        uintptr_t current_address = (uintptr_t)(arena->base_ptr + current_offset);
        uintptr_t aligned_address = vdb_align_forward(current_address, alignment);
        aligned_offset = current_offset + (aligned_address - current_address);
        new_offset = aligned_offset + size;

        if (new_offset > arena->capacity) {
            // OOM condition met within arena boundaries. Enforce deterministic backpressure.
            return VDB_ALLOC_ERR_OOM;
        }

        if (atomic_compare_exchange_weak_explicit(&arena->offset,
                                                  &current_offset,
                                                  new_offset,
                                                  memory_order_acq_rel,
                                                  memory_order_relaxed)) {
            break;
        }
    }

    *out_ptr = (void*)(arena->base_ptr + aligned_offset);
    return VDB_ALLOC_SUCCESS;
}

void vdb_arena_reset(vdb_arena_t *arena) {
    if (!arena || arena->magic != VECTOR_DB_ARENA_MAGIC) {
        return;
    }
    atomic_store_explicit(&arena->offset, 0, memory_order_release);
}

vdb_alloc_status_t vdb_thread_cache_init(void) {
    if (atomic_load_explicit(&tls_initialized, memory_order_acquire)) {
        return VDB_ALLOC_SUCCESS;
    }

    vdb_arena_t *arena = NULL;
    vdb_alloc_status_t status = vdb_arena_create(&arena, 1024 * 1024 * 32, -1); // 32MB thread local arena
    if (status != VDB_ALLOC_SUCCESS) {
        return status;
    }

    arena->is_thread_local = true;
    tls_cache.local_arena = arena;
    tls_cache.allocations_count = 0;
    tls_cache.bytes_allocated = 0;

    atomic_store_explicit(&tls_initialized, true, memory_order_release);
    return VDB_ALLOC_SUCCESS;
}

void vdb_thread_cache_cleanup(void) {
    if (!atomic_load_explicit(&tls_initialized, memory_order_acquire)) {
        return;
    }

    if (tls_cache.local_arena) {
        vdb_arena_destroy(tls_cache.local_arena);
        tls_cache.local_arena = NULL;
    }
    tls_cache.allocations_count = 0;
    tls_cache.bytes_allocated = 0;

    atomic_store_explicit(&tls_initialized, false, memory_order_release);
}

vdb_alloc_status_t vdb_thread_local_allocate(size_t size, size_t alignment, void **out_ptr) {
    if (!atomic_load_explicit(&tls_initialized, memory_order_acquire)) {
        vdb_alloc_status_t init_status = vdb_thread_cache_init();
        if (init_status != VDB_ALLOC_SUCCESS) {
            return init_status;
        }
    }

    if (!tls_cache.local_arena) {
        return VDB_ALLOC_ERR_THREAD_LOCAL;
    }

    vdb_alloc_status_t status = vdb_arena_allocate(tls_cache.local_arena, size, alignment, out_ptr);
    if (status == VDB_ALLOC_SUCCESS) {
        tls_cache.allocations_count++;
        tls_cache.bytes_allocated += size;
    }
    return status;
}

#endif /* VECTOR_DB_ALLOCATOR_C */