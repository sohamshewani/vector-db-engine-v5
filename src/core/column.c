#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

#define CACHE_LINE_SIZE 64
#define CHUNK_CAPACITY 8192
#define MAX_DICTIONARY_STRINGS 65536
#define MAX_STRING_LENGTH 256

typedef enum {
    COLUMN_TYPE_INT32 = 0,
    COLUMN_TYPE_FLOAT32 = 1,
    COLUMN_TYPE_STRING = 2
} ColumnType;

typedef enum {
    ENCODING_RAW = 0,
    ENCODING_DICTIONARY = 1,
    ENCODING_RLE = 2
} EncodingType;

typedef struct {
    uint8_t* data;
    size_t capacity;
    size_t size;
    _Atomic size_t ref_count;
} MemoryArena;

typedef struct {
    char** strings;
    uint32_t* hash_table;
    size_t count;
    size_t capacity;
    MemoryArena* string_arena;
} Dictionary;

typedef struct {
    uint32_t chunk_id;
    ColumnType type;
    EncodingType encoding;
    size_t row_count;
    
    // 64-byte aligned data pointers
    void* values;             // SoA values vector
    uint64_t* null_bitmap;    // Null bit vector
    
    // Dictionary specific
    Dictionary* dict;
    void* encoded_ids;        // uint8_t, uint16_t, or uint32_t IDs
    
    // RLE specific
    uint32_t* rle_runs;
    void* rle_values;
    
    _Atomic uint32_t version;
    _Atomic int32_t ref_count;
    bool is_sealed;
    
    struct ColumnChunk* next;
} __attribute__((aligned(CACHE_LINE_SIZE))) ColumnChunk;

typedef struct {
    char name[64];
    ColumnType type;
    _Atomic(ColumnChunk*) head_chunk;
    MemoryArena* arena;
    _Atomic uint32_t metadata_version;
} Column;

static inline void* cache_aligned_alloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, CACHE_LINE_SIZE, size) != 0) {
        return NULL;
    }
    return ptr;
}

static inline void cache_aligned_free(void* ptr) {
    free(ptr);
}

MemoryArena* memory_arena_create(size_t capacity) {
    MemoryArena* arena = (MemoryArena*)cache_aligned_alloc(sizeof(MemoryArena));
    if (!arena) return NULL;
    
    size_t aligned_capacity = (capacity + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    arena->data = (uint8_t*)cache_aligned_alloc(aligned_capacity);
    if (!arena->data) {
        cache_aligned_free(arena);
        return NULL;
    }
    
    arena->capacity = aligned_capacity;
    arena->size = 0;
    atomic_init(&arena->ref_count, 1);
    return arena;
}

void memory_arena_destroy(MemoryArena* arena) {
    if (!arena) return;
    if (atomic_fetch_sub(&arena->ref_count, 1) <= 1) {
        cache_aligned_free(arena->data);
        cache_aligned_free(arena);
    }
}

void* memory_arena_alloc(MemoryArena* arena, size_t size) {
    size_t aligned_size = (size + 7) & ~7; // 8-byte alignment inside arena
    size_t current_size = atomic_load(&arena->size);
    
    while (1) {
        if (current_size + aligned_size > arena->capacity) {
            return NULL; // OOM backpressure trigger
        }
        if (atomic_compare_exchange_weak(&arena->size, &current_size, current_size + aligned_size)) {
            return arena->data + current_size;
        }
    }
}

Column* column_create(const char* name, ColumnType type) {
    Column* col = (Column*)cache_aligned_alloc(sizeof(Column));
    if (!col) return NULL;
    
    strncpy(col->name, name, sizeof(col->name) - 1);
    col->name[sizeof(col->name) - 1] = '\0';
    col->type = type;
    
    col->arena = memory_arena_create(1024 * 1024 * 16); // 16MB initial arena
    if (!col->arena) {
        cache_aligned_free(col);
        return NULL;
    }
    
    atomic_init(&col->head_chunk, NULL);
    atomic_init(&col->metadata_version, 0);
    return col;
}

void column_destroy(Column* col) {
    if (!col) return;
    
    ColumnChunk* chunk = atomic_load(&col->head_chunk);
    while (chunk) {
        ColumnChunk* next = chunk->next;
        if (chunk->values) cache_aligned_free(chunk->values);
        if (chunk->null_bitmap) cache_aligned_free(chunk->null_bitmap);
        if (chunk->encoded_ids) cache_aligned_free(chunk->encoded_ids);
        if (chunk->rle_runs) cache_aligned_free(chunk->rle_runs);
        if (chunk->rle_values) cache_aligned_free(chunk->rle_values);
        if (chunk->dict) {
            cache_aligned_free(chunk->dict->strings);
            cache_aligned_free(chunk->dict->hash_table);
            cache_aligned_free(chunk->dict);
        }
        cache_aligned_free(chunk);
        chunk = next;
    }
    
    memory_arena_destroy(col->arena);
    cache_aligned_free(col);
}

ColumnChunk* column_chunk_create(ColumnType type, uint32_t chunk_id) {
    ColumnChunk* chunk = (ColumnChunk*)cache_aligned_alloc(sizeof(ColumnChunk));
    if (!chunk) return NULL;
    
    memset(chunk, 0, sizeof(ColumnChunk));
    chunk->chunk_id = chunk_id;
    chunk->type = type;
    chunk->encoding = ENCODING_RAW;
    chunk->row_count = 0;
    chunk->is_sealed = false;
    atomic_init(&chunk->version, 0);
    atomic_init(&chunk->ref_count, 1);
    
    size_t value_size = (type == COLUMN_TYPE_INT32 || type == COLUMN_TYPE_FLOAT32) ? sizeof(float) : sizeof(uint64_t);
    chunk->values = cache_aligned_alloc(CHUNK_CAPACITY * value_size);
    chunk->null_bitmap = cache_aligned_alloc(((CHUNK_CAPACITY + 63) / 64) * sizeof(uint64_t));
    
    if (!chunk->values || !chunk->null_bitmap) {
        if (chunk->values) cache_aligned_free(chunk->values);
        if (chunk->null_bitmap) cache_aligned_free(chunk->null_bitmap);
        cache_aligned_free(chunk);
        return NULL;
    }
    
    memset(chunk->null_bitmap, 0xFF, ((CHUNK_CAPACITY + 63) / 64) * sizeof(uint64_t)); // Default all valid (1)
    return chunk;
}

bool column_append_int32(Column* col, int32_t value, bool is_null) {
    ColumnChunk* head = atomic_load(&col->head_chunk);
    
    if (!head || head->is_sealed || head->row_count >= CHUNK_CAPACITY) {
        uint32_t new_id = head ? head->chunk_id + 1 : 1;
        ColumnChunk* new_chunk = column_chunk_create(COLUMN_TYPE_INT32, new_id);
        if (!new_chunk) return false;
        
        do {
            new_chunk->next = atomic_load(&col->head_chunk);
        } while (!atomic_compare_exchange_weak(&col->head_chunk, &new_chunk->next, new_chunk));
        
        head = new_chunk;
    }
    
    size_t idx = head->row_count;
    ((int32_t*)head->values)[idx] = value;
    
    if (!is_null) {
        head->null_bitmap[idx / 64] &= ~(1ULL << (idx % 64));
    } else {
        head->null_bitmap[idx / 64] |= (1ULL << (idx % 64));
    }
    
    head->row_count++;
    atomic_fetch_add(&head->version, 1);
    return true;
}

bool column_append_float32(Column* col, float value, bool is_null) {
    ColumnChunk* head = atomic_load(&col->head_chunk);
    
    if (!head || head->is_sealed || head->row_count >= CHUNK_CAPACITY) {
        uint32_t new_id = head ? head->chunk_id + 1 : 1;
        ColumnChunk* new_chunk = column_chunk_create(COLUMN_TYPE_FLOAT32, new_id);
        if (!new_chunk) return false;
        
        do {
            new_chunk->next = atomic_load(&col->head_chunk);
        } while (!atomic_compare_exchange_weak(&col->head_chunk, &new_chunk->next, new_chunk));
        
        head = new_chunk;
    }
    
    size_t idx = head->row_count;
    ((float*)head->values)[idx] = value;
    
    if (!is_null) {
        head->null_bitmap[idx / 64] &= ~(1ULL << (idx % 64));
    } else {
        head->null_bitmap[idx / 64] |= (1ULL << (idx % 64));
    }
    
    head->row_count++;
    atomic_fetch_add(&head->version, 1);
    return true;
}

void column_chunk_seal(ColumnChunk* chunk) {
    if (!chunk || chunk->is_sealed) return;
    chunk->is_sealed = true;
    atomic_fetch_add(&chunk->version, 1);
}

void column_vector_scan_int32(const ColumnChunk* chunk, int32_t* out_values, uint64_t* out_null_bitmap, size_t* out_count) {
    if (!chunk || chunk->type != COLUMN_TYPE_INT32) {
        *out_count = 0;
        return;
    }
    
    size_t count = chunk->row_count;
    *out_count = count;
    
    if (chunk->encoding == ENCODING_RAW) {
        memcpy(out_values, chunk->values, count * sizeof(int32_t));
        memcpy(out_null_bitmap, chunk->null_bitmap, ((count + 63) / 64) * sizeof(uint64_t));
    } else if (chunk->encoding == ENCODING_RLE) {
        size_t current_row = 0;
        uint32_t* runs = chunk->rle_runs;
        int32_t* rle_vals = (int32_t*)chunk->rle_values;
        
        // Find total runs stored safely or track via metadata
        // For production completeness, we decode RLE inline
        for (int i = 0; current_row < count; ++i) {
            uint32_t run_length = runs[i];
            int32_t val = rle_vals[i];
            for (uint32_t r = 0; r < run_length && current_row < count; ++r, ++current_row) {
                out_values[current_row] = val;
            }
        }
        memcpy(out_null_bitmap, chunk->null_bitmap, ((count + 63) / 64) * sizeof(uint64_t));
    }
}

void column_compress_rle_int32(ColumnChunk* chunk) {
    if (!chunk || chunk->type != COLUMN_TYPE_INT32 || chunk->encoding != ENCODING_RAW || !chunk->is_sealed) {
        return;
    }
    
    size_t count = chunk->row_count;
    if (count == 0) return;
    
    int32_t* raw_vals = (int32_t*)chunk->values;
    uint32_t* temp_runs = (uint32_t*)cache_aligned_alloc(count * sizeof(uint32_t));
    int32_t* temp_vals = (int32_t*)cache_aligned_alloc(count * sizeof(int32_t));
    
    if (!temp_runs || !temp_vals) {
        if (temp_runs) cache_aligned_free(temp_runs);
        if (temp_vals) cache_aligned_free(temp_vals);
        return; // OOM graceful fallback
    }
    
    size_t run_count = 0;
    int32_t current_val = raw_vals[0];
    uint32_t current_run = 1;
    
    for (size_t i = 1; i < count; ++i) {
        if (raw_vals[i] == current_val && current_run < UINT32_MAX) {
            current_run++;
        } else {
            temp_runs[run_count] = current_run;
            temp_vals[run_count] = current_val;
            run_count++;
            current_val = raw_vals[i];
            current_run = 1;
        }
    }
    temp_runs[run_count] = current_run;
    temp_vals[run_count] = current_val;
    run_count++;
    
    chunk->rle_runs = (uint32_t*)cache_aligned_alloc(run_count * sizeof(uint32_t));
    chunk->rle_values = (int32_t*)cache_aligned_alloc(run_count * sizeof(int32_t));
    
    if (chunk->rle_runs && chunk->rle_values) {
        memcpy(chunk->rle_runs, temp_runs, run_count * sizeof(uint32_t));
        memcpy(chunk->rle_values, temp_vals, run_count * sizeof(int32_t));
        
        cache_aligned_free(chunk->values);
        chunk->values = NULL;
        chunk->encoding = ENCODING_RLE;
        atomic_fetch_add(&chunk->version, 1);
    }
    
    cache_aligned_free(temp_runs);
    cache_aligned_free(temp_vals);
}

void column_vector_filter_gt_int32(const ColumnChunk* chunk, int32_t threshold, uint64_t* restrict out_mask) {
    if (!chunk || chunk->type != COLUMN_TYPE_INT32) return;
    
    size_t count = chunk->row_count;
    int32_t* vals = (int32_t*)chunk->values;
    
    if (chunk->encoding == ENCODING_RAW) {
        size_t i = 0;
        // Vectorized AVX2 comparison path
        #__AVX2__ // Evaluated statically by platform assumption in production
        for (; i + 7 < count; i += 8) {
            __m256i v = _mm256_loadu_si256((__m256i*)&vals[i]);
            __m256i t = _mm256_set1_epi32(threshold);
            __m256i cmp = _mm256_cmpgt_epi32(v, t);
            int mask = _mm256_movemask