#ifndef VECTOR_DB_TYPES_H
#define VECTOR_DB_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <string.h>

#define VEC_DB_CACHE_LINE 64
#define VEC_DB_VECTOR_WIDTH 256
#define VEC_DB_MAX_STRING_LEN 255

typedef enum {
    VEC_DB_SUCCESS = 0,
    VEC_DB_ERR_OOM = -1,
    VEC_DB_ERR_INVALID_ARG = -2,
    VEC_DB_ERR_BOUNDS = -3,
    VEC_DB_ERR_MUTEX = -4,
    VEC_DB_ERR_CORRUPTION = -5,
    VEC_DB_ERR_BUSY = -6
} vec_db_status_t;

typedef enum {
    VEC_DB_TYPE_INT8 = 1,
    VEC_DB_TYPE_INT16 = 2,
    VEC_DB_TYPE_INT32 = 3,
    VEC_DB_TYPE_INT64 = 4,
    VEC_DB_TYPE_FLOAT32 = 5,
    VEC_DB_TYPE_FLOAT64 = 6,
    VEC_DB_TYPE_DICTIONARY = 7,
    VEC_DB_TYPE_RLE = 8
} vec_db_data_type_t;

typedef enum {
    VEC_DB_CHUNK_MUTABLE = 0,
    VEC_DB_CHUNK_SEALED = 1,
    VEC_DB_CHUNK_COMPACTED = 2
} vec_db_chunk_state_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    uint64_t epoch;
    atomic_bool active;
    void* ptr;
} vec_db_hazard_pointer_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    atomic_uint_fast64_t global_epoch;
    vec_db_hazard_pointer_t hp_table[64];
    atomic_size_t hp_count;
} vec_db_egcc_domain_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    uint8_t* base;
    size_t capacity;
    atomic_size_t offset;
    bool is_thread_local;
} vec_db_arena_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    void* data;
    uint64_t* null_bitmap;
    size_t capacity;
    size_t length;
    vec_db_data_type_t type;
} vec_db_column_vector_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    char** strings;
    uint32_t* hash_table;
    size_t count;
    size_t capacity;
    vec_db_arena_t string_arena;
} vec_db_dictionary_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    uint32_t* runs;
    void* values;
    size_t run_count;
    vec_db_data_type_t value_type;
} vec_db_rle_segment_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    uint64_t* words;
    size_t word_count;
    size_t cardinality;
    bool is_dense;
} vec_db_bitmap_index_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    uint64_t chunk_id;
    atomic_int state;
    atomic_uint_fast64_t version_stamp;
    vec_db_column_vector_t* columns;
    size_t column_count;
    size_t row_count;
    vec_db_bitmap_index_t primary_index;
} vec_db_chunk_t;

typedef struct alignas(VEC_DB_CACHE_LINE) {
    vec_db_chunk_t** chunks;
    atomic_size_t chunk_count;
    size_t chunk_capacity;
    atomic_uint_fast64_t metadata_version;
    vec_db_egcc_domain_t egcc;
} vec_db_table_t;

static inline vec_db_status_t vec_db_arena_init(vec_db_arena_t* arena, size_t capacity, bool thread_local) {
    if (!arena || capacity == 0) return VEC_DB_ERR_INVALID_ARG;
    
    posix_memalign((void**)&arena->base, VEC_DB_CACHE_LINE, capacity);
    if (!arena->base) return VEC_DB_ERR_OOM;
    
    arena->capacity = capacity;
    atomic_init(&arena->offset, 0);
    arena->is_thread_local = thread_local;
    memset(arena->base, 0, capacity);
    return VEC_DB_SUCCESS;
}

static inline void vec_db_arena_destroy(vec_db_arena_t* arena) {
    if (arena && arena->base) {
        free(arena->base);
        arena->base = NULL;
        arena->capacity = 0;
        atomic_store(&arena->offset, 0);
    }
}

static inline void* vec_db_arena_alloc(vec_db_arena_t* arena, size_t size) {
    if (!arena || size == 0) return NULL;
    
    size_t aligned_size = (size + (VEC_DB_CACHE_LINE - 1)) & ~(VEC_DB_CACHE_LINE - 1);
    size_t current = atomic_load(&arena->offset);
    
    while (1) {
        if (current + aligned_size > arena->capacity) {
            return NULL; // OOM backpressure trigger
        }
        if (atomic_compare_exchange_weak(&arena->offset, &current, current + aligned_size)) {
            return (void*)(arena->base + current);
        }
    }
}

static inline vec_db_status_t vec_db_column_vector_init(vec_db_column_vector_t* col, vec_db_data_type_t type, size_t capacity, vec_db_arena_t* arena) {
    if (!col || capacity == 0) return VEC_DB_ERR_INVALID_ARG;
    
    size_t type_size = 0;
    switch (type) {
        case VEC_DB_TYPE_INT8: type_size = sizeof(int8_t); break;
        case VEC_DB_TYPE_INT16: type_size = sizeof(int16_t); break;
        case VEC_DB_TYPE_INT32: type_size = sizeof(int32_t); break;
        case VEC_DB_TYPE_INT64: type_size = sizeof(int64_t); break;
        case VEC_DB_TYPE_FLOAT32: type_size = sizeof(float); break;
        case VEC_DB_TYPE_FLOAT64: type_size = sizeof(double); break;
        case VEC_DB_TYPE_DICTIONARY: type_size = sizeof(uint32_t); break;
        case VEC_DB_TYPE_RLE: type_size = sizeof(uint64_t); break;
        default: return VEC_DB_ERR_INVALID_ARG;
    }
    
    col->data = vec_db_arena_alloc(arena, capacity * type_size);
    if (!col->data) return VEC_DB_ERR_OOM;
    
    size_t bitmap_size = ((capacity + 63) / 64) * sizeof(uint64_t);
    col->null_bitmap = vec_db_arena_alloc(arena, bitmap_size);
    if (!col->null_bitmap) return VEC_DB_ERR_OOM;
    
    col->capacity = capacity;
    col->length = 0;
    col->type = type;
    memset(col->null_bitmap, 0xFF, bitmap_size); // Default all non-null
    return VEC_DB_SUCCESS;
}

static inline vec_db_status_t vec_db_dictionary_init(vec_db_dictionary_t* dict, size_t capacity, size_t arena_size) {
    if (!dict || capacity == 0) return VEC_DB_ERR_INVALID_ARG;
    
    posix_memalign((void**)&dict->strings, VEC_DB_CACHE_LINE, capacity * sizeof(char*));
    posix_memalign((void**)&dict->hash_table, VEC_DB_CACHE_LINE, capacity * sizeof(uint32_t));
    
    if (!dict->strings || !dict->hash_table) {
        free(dict->strings);
        free(dict->hash_table);
        return VEC_DB_ERR_OOM;
    }
    
    vec_db_status_t status = vec_db_arena_init(&dict->string_arena, arena_size, false);
    if (status != VEC_DB_SUCCESS) {
        free(dict->strings);
        free(dict->hash_table);
        return status;
    }
    
    dict->count = 0;
    dict->capacity = capacity;
    memset(dict->hash_table, 0xFF, capacity * sizeof(uint32_t));
    return VEC_DB_SUCCESS;
}

static inline void vec_db_dictionary_destroy(vec_db_dictionary_t* dict) {
    if (dict) {
        free(dict->strings);
        free(dict->hash_table);
        vec_db_arena_destroy(&dict->string_arena);
        dict->strings = NULL;
        dict->hash_table = NULL;
        dict->count = 0;
        dict->capacity = 0;
    }
}

static inline uint32_t vec_db_hash_string(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint32_t)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static inline int32_t vec_db_dictionary_get_or_insert(vec_db_dictionary_t* dict, const char* str) {
    if (!dict || !str) return VEC_DB_ERR_INVALID_ARG;
    
    uint32_t hash = vec_db_hash_string(str);
    uint32_t idx = hash % dict->capacity;
    
    while (dict->hash_table[idx] != UINT32_MAX) {
        uint32_t existing_idx = dict->hash_table[idx];
        if (strcmp(dict->strings[existing_idx], str) == 0) {
            return (int32_t)existing_idx;
        }
        idx = (idx + 1) % dict->capacity;
    }
    
    if (dict->count >= dict->capacity) {
        return VEC_DB_ERR_OOM;
    }
    
    size_t len = strlen(str) + 1;
    char* dest = (char*)vec_db_arena_alloc(&dict->string_arena, len);
    if (!dest) return VEC_DB_ERR_OOM;
    
    memcpy(dest, str, len);
    
    uint32_t new_idx = (uint32_t)dict->count++;
    dict->strings[new_idx] = dest;
    dict->hash_table[idx] = new_idx;
    
    return (int32_t)new_idx;
}

static inline vec_db_status_t vec_db_bitmap_init(vec_db_bitmap_index_t* bitmap, size_t bit_capacity) {
    if (!bitmap || bit_capacity == 0) return VEC_DB_ERR_INVALID_ARG;
    
    size_t word_count = (bit_capacity + 63) / 64;
    posix_memalign((void**)&bitmap->words, VEC_DB_CACHE_LINE, word_count * sizeof(uint64_t));
    if (!bitmap->words) return VEC_DB_ERR_OOM;
    
    memset(bitmap->words, 0, word_count * sizeof(uint64_t));
    bitmap->word_count = word_count;
    bitmap->cardinality = 0;
    bitmap->is_dense = false;
    return VEC_DB_SUCCESS;
}

static inline void vec_db_bitmap_destroy(vec_db_bitmap_index_t* bitmap) {
    if (bitmap && bitmap->words) {
        free(bitmap->words);
        bitmap->words = NULL;
        bitmap->word_count = 0;
        bitmap->cardinality = 0;
    }
}

static inline void vec_db_bitmap_set(vec_db_bitmap_index_t* bitmap, size_t bit_index) {
    if (!bitmap) return;
    size_t word_idx = bit_index / 64;
    if (word_idx >= bitmap->word_count) return;
    
    uint64_t mask = 1ULL << (bit_index % 64);
    if (!(bitmap->words[word_idx] & mask)) {
        bitmap->words[word_idx] |= mask;
        bitmap->cardinality++;
    }
}

static inline bool vec_db_bitmap_get(const vec_db_bitmap_index_t* bitmap, size_t bit_index) {
    if (!bitmap) return false;
    size_t word_idx = bit_index / 64;
    if (word_idx >= bitmap->word_count) return false;
    
    return (bitmap->words[word_idx] & (1ULL << (bit_index % 64))) != 0;
}

static inline vec_db_status_t vec_db_chunk_init(vec_db_chunk_t* chunk, uint64_t chunk_id, size_t column_count, size_t row_capacity, vec_db_arena_t* arena) {
    if (!chunk || column_count == 0 || row_capacity == 0) return VEC_DB_ERR_INVALID_ARG;
    
    chunk->chunk_id = chunk_id;
    atomic_init(&chunk->state, VEC_DB_CHUNK_MUTABLE);
    atomic_init(&chunk->version_stamp, 1);
    chunk->row_count = 0;
    chunk->column_count = column_count;
    
    chunk->columns = vec_db_arena_alloc(arena, column_count * sizeof(vec_db_column_vector_t));
    if (!chunk->columns) return VEC_DB_ERR_OOM;
    
    for (size_t i = 0; i < column_count; i++) {
        vec_db_status_t status = vec_db_column_vector_init(&chunk->columns[i], VEC_DB_TYPE_INT64, row_capacity, arena);
        if (status != VEC_DB_SUCCESS) return status;
    }
    
    return vec_db_bitmap_init(&chunk->primary_index, row_capacity);
}

static inline void vec_db_chunk_destroy(vec_db_chunk_t* chunk) {
    if (chunk) {
        vec_db_bitmap_destroy(&chunk->primary_index);
        chunk->columns = NULL;
    }
}

static inline void vec_db_egcc_register(vec_db_egcc_domain_t* domain, uint32_t thread_id, vec_db_hazard_pointer_t** hp) {
    if (!domain || thread_id >= 64) return;
    domain->hp_table[thread_id].active = true;
    domain->hp_table[thread_id].epoch = atomic_load(&domain->global_epoch);
    *hp = &domain->hp_table[thread_id];
}

static inline void vec_db_egcc_retire(vec_db_egcc_domain_t* domain, void* ptr) {
    uint64_t current_epoch = atomic_fetch_add(&domain