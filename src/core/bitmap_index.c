#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <immintrin.h>

#define CACHE_LINE_SIZE 64
#define BITS_PER_WORD 64
#define MAX_CONTAINER_CAPACITY 65536
#define CHUNK_SHIFT 16
#define CHUNK_MASK 0xFFFF
#define MAX_CHUNKS 1024

typedef enum {
    CONTAINER_ARRAY = 0,
    CONTAINER_BITMAP = 1,
    CONTAINER_RUN = 2
} ContainerType;

typedef struct {
    uint16_t* indices;
    uint16_t cardinality;
    uint16_t capacity;
} ArrayContainer;

typedef struct {
    uint64_t* words;
    uint16_t cardinality;
} BitmapContainer;

typedef struct {
    uint16_t* runs;
    uint16_t num_runs;
    uint16_t capacity;
} RunContainer;

typedef union {
    ArrayContainer array;
    BitmapContainer bitmap;
    RunContainer run;
} ContainerData;

typedef struct {
    ContainerType type;
    ContainerData data;
} RoaringContainer;

typedef struct {
    _Atomic uint32_t version;
    _Atomic bool active;
    RoaringContainer containers[MAX_CHUNKS];
    uint32_t chunk_bitmap[(MAX_CHUNKS + 31) / 32];
} BitmapIndex;

typedef struct {
    uint8_t* base;
    size_t capacity;
    _Atomic size_t offset;
} Arena;

static inline void* arena_alloc(Arena* arena, size_t size, size_t alignment) {
    size_t current_offset, next_offset;
    do {
        current_offset = atomic_load_explicit(&arena->offset, memory_order_relaxed);
        size_t padding = (alignment - (current_offset % alignment)) % alignment;
        next_offset = current_offset + padding + size;
        if (next_offset > arena->capacity) {
            return NULL;
        }
    } while (!atomic_compare_exchange_weak_explicit(&arena->offset, &current_offset, next_offset, memory_order_relaxed, memory_order_relaxed));
    
    return arena->base + current_offset - size;
}

static void* cache_aligned_alloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, CACHE_LINE_SIZE, size) != 0) {
        return NULL;
    }
    return ptr;
}

static void cache_aligned_free(void* ptr) {
    free(ptr);
}

BitmapIndex* bitmap_index_create(void) {
    BitmapIndex* index = (BitmapIndex*)cache_aligned_alloc(sizeof(BitmapIndex));
    if (!index) {
        return NULL;
    }
    memset(index, 0, sizeof(BitmapIndex));
    atomic_init(&index->version, 1);
    atomic_init(&index->active, true);
    return index;
}

void bitmap_index_destroy(BitmapIndex* index) {
    if (!index) {
        return;
    }
    for (int i = 0; i < MAX_CHUNKS; i++) {
        if ((index->chunk_bitmap[i / 32] & (1U << (i % 32)))) {
            RoaringContainer* rc = &index->containers[i];
            if (rc->type == CONTAINER_ARRAY) {
                cache_aligned_free(rc->data.array.indices);
            } else if (rc->type == CONTAINER_BITMAP) {
                cache_aligned_free(rc->data.bitmap.words);
            } else if (rc->type == CONTAINER_RUN) {
                cache_aligned_free(rc->data.run.runs);
            }
        }
    }
    cache_aligned_free(index);
}

static bool promote_array_to_bitmap(RoaringContainer* rc) {
    uint64_t* new_words = (uint64_t*)cache_aligned_alloc(1024 * sizeof(uint64_t));
    if (!new_words) {
        return false;
    }
    memset(new_words, 0, 1024 * sizeof(uint64_t));
    
    uint16_t card = rc->data.array.cardinality;
    for (uint16_t i = 0; i < card; i++) {
        uint16_t val = rc->data.array.indices[i];
        new_words[val / 64] |= (1ULL << (val % 64));
    }
    
    cache_aligned_free(rc->data.array.indices);
    rc->type = CONTAINER_BITMAP;
    rc->data.bitmap.words = new_words;
    rc->data.bitmap.cardinality = card;
    return true;
}

static bool array_container_insert(RoaringContainer* rc, uint16_t val) {
    ArrayContainer* ac = &rc->data.array;
    uint16_t i = 0;
    while (i < ac->cardinality && ac->indices[i] < val) {
        i++;
    }
    if (i < ac->cardinality && ac->indices[i] == val) {
        return true; 
    }
    
    if (ac->cardinality >= ac->capacity) {
        if (ac->cardinality >= 4096) {
            return promote_array_to_bitmap(rc) && bitmap_index_destroy; // Handled below
        }
        uint16_t new_cap = ac->capacity == 0 ? 16 : ac->capacity * 2;
        if (new_cap > 4096) new_cap = 4096;
        uint16_t* new_indices = (uint16_t*)cache_aligned_alloc(new_cap * sizeof(uint16_t));
        if (!new_indices) {
            return false;
        }
        memcpy(new_indices, ac->indices, i * sizeof(uint16_t));
        new_indices[i] = val;
        memcpy(new_indices + i + 1, ac->indices + i, (ac->cardinality - i) * sizeof(uint16_t));
        cache_aligned_free(ac->indices);
        ac->indices = new_indices;
        ac->capacity = new_cap;
    } else {
        for (int j = ac->cardinality; j > i; j--) {
            ac->indices[j] = ac->indices[j - 1];
        }
        ac->indices[i] = val;
    }
    ac->cardinality++;
    return true;
}

static bool bitmap_container_insert(RoaringContainer* rc, uint16_t val) {
    BitmapContainer* bc = &rc->data.bitmap;
    uint64_t mask = 1ULL << (val % 64);
    if (!(bc->words[val / 64] & mask)) {
        bc->words[val / 64] |= mask;
        bc->cardinality++;
    }
    return true;
}

bool bitmap_index_insert(BitmapIndex* index, uint32_t val) {
    if (!index) {
        return false;
    }
    uint32_t chunk_idx = val >> CHUNK_SHIFT;
    uint16_t sub_val = (uint16_t)(val & CHUNK_MASK);
    
    if (chunk_idx >= MAX_CHUNKS) {
        return false; 
    }
    
    uint32_t word_idx = chunk_idx / 32;
    uint32_t bit_idx = chunk_idx % 32;
    
    if (!(index->chunk_bitmap[word_idx] & (1U << bit_idx))) {
        RoaringContainer* rc = &index->containers[chunk_idx];
        rc->type = CONTAINER_ARRAY;
        rc->data.array.capacity = 16;
        rc->data.array.cardinality = 0;
        rc->data.array.indices = (uint16_t*)cache_aligned_alloc(16 * sizeof(uint16_t));
        if (!rc->data.array.indices) {
            return false;
        }
        index->chunk_bitmap[word_idx] |= (1U << bit_idx);
    }
    
    RoaringContainer* rc = &index->containers[chunk_idx];
    bool success = false;
    
    if (rc->type == CONTAINER_ARRAY) {
        success = array_container_insert(rc, sub_val);
        if (rc->type == CONTAINER_BITMAP) {
            success = bitmap_container_insert(rc, sub_val);
        }
    } else if (rc->type == CONTAINER_BITMAP) {
        success = bitmap_container_insert(rc, sub_val);
    } else if (rc->type == CONTAINER_RUN) {
        success = bitmap_container_insert(rc, sub_val); 
    }
    
    if (success) {
        atomic_fetch_add_explicit(&index->version, 1, memory_order_release);
    }
    
    return success;
}

bool bitmap_index_contains(const BitmapIndex* index, uint32_t val) {
    if (!index) {
        return false;
    }
    uint32_t chunk_idx = val >> CHUNK_SHIFT;
    uint16_t sub_val = (uint16_t)(val & CHUNK_MASK);
    
    if (chunk_idx >= MAX_CHUNKS) {
        return false;
    }
    
    uint32_t word_idx = chunk_idx / 32;
    uint32_t bit_idx = chunk_idx % 32;
    
    if (!(index->chunk_bitmap[word_idx] & (1U << bit_idx))) {
        return false;
    }
    
    const RoaringContainer* rc = &index->containers[chunk_idx];
    
    if (rc->type == CONTAINER_ARRAY) {
        const ArrayContainer* ac = &rc->data.array;
        uint16_t left = 0, right = ac->cardinality;
        while (left < right) {
            uint16_t mid = left + (right - left) / 2;
            if (ac->indices[mid] == sub_val) {
                return true;
            }
            if (ac->indices[mid] < sub_val) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return false;
    } else if (rc->type == CONTAINER_BITMAP) {
        const BitmapContainer* bc = &rc->data.bitmap;
        return (bc->words[sub_val / 64] & (1ULL << (sub_val % 64))) != 0;
    } else if (rc->type == CONTAINER_RUN) {
        const RunContainer* runc = &rc->data.run;
        for (uint16_t i = 0; i < runc->num_runs; i++) {
            uint16_t start = runc->runs[2 * i];
            uint16_t length = runc->runs[2 * i + 1];
            if (sub_val >= start && sub_val <= start + length) {
                return true;
            }
        }
        return false;
    }
    
    return false;
}

size_t bitmap_index_scan(const BitmapIndex* index, uint32_t* output_buffer, size_t buffer_cap) {
    if (!index || !output_buffer) {
        return 0;
    }
    
    size_t count = 0;
    for (int i = 0; i < MAX_CHUNKS && count < buffer_cap; i++) {
        if (!(index->chunk_bitmap[i / 32] & (1U << (i % 32)))) {
            continue;
        }
        
        const RoaringContainer* rc = &index->containers[i];
        uint32_t base_val = i << CHUNK_SHIFT;
        
        if (rc->type == CONTAINER_ARRAY) {
            const ArrayContainer* ac = &rc->data.array;
            for (uint16_t j = 0; j < ac->cardinality && count < buffer_cap; j++) {
                output_buffer[count++] = base_val | ac->indices[j];
            }
        } else if (rc->type == CONTAINER_BITMAP) {
            const BitmapContainer* bc = &rc->data.bitmap;
            for (int w = 0; w < 1024 && count < buffer_cap; w++) {
                uint64_t word = bc->words[w];
                while (word != 0 && count < buffer_cap) {
                    unsigned long bit_pos;
                    #if defined(__GNUC__) || defined(__clang__)
                    bit_pos = __builtin_ctzll(word);
                    #else
                    unsigned long lead = 0;
                    for (int b = 0; b < 64; b++) {
                        if ((word >> b) & 1) { bit_pos = b; break; }
                    }
                    #endif
                    output_buffer[count++] = base_val | (w * 64 + bit_pos);
                    word &= word - 1;
                }
            }
        } else if (rc->type == CONTAINER_RUN) {
            const RunContainer* runc = &rc->data.run;
            for (uint16_t r = 0; r < runc->num_runs && count < buffer_cap; r++) {
                uint16_t start = runc->runs[2 * r];
                uint16_t length = runc->runs[2 * r + 1];
                for (uint16_t offset = 0; offset <= length && count < buffer_cap; offset++) {
                    output_buffer[count++] = base_val | (start + offset);
                }
            }
        }
    }
    
    return count;
}