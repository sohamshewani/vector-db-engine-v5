#include "core/vector_exec.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <immintrin.h>
#include <pthread.h>
#include <errno.h>

#define CACHE_LINE_SIZE 64
#define VECTOR_WIDTH 8
#define MAX_THREADS 64

typedef struct {
    uint8_t* base_ptr;
    size_t capacity;
    atomic_size_t offset;
} ThreadArena;

static __thread ThreadArena* current_thread_arena = NULL;

static void* arena_alloc(size_t size) {
    size_t aligned_size = (size + (CACHE_LINE_SIZE - 1)) & ~(CACHE_LINE_SIZE - 1);
    if (!current_thread_arena) {
        current_thread_arena = (ThreadArena*)malloc(sizeof(ThreadArena));
        if (!current_thread_arena) return NULL;
        current_thread_arena->capacity = 1024 * 1024 * 16; 
        current_thread_arena->base_ptr = (uint8_t*)aligned_alloc(CACHE_LINE_SIZE, current_thread_arena->capacity);
        if (!current_thread_arena->base_ptr) {
            free(current_thread_arena);
            current_thread_arena = NULL;
            return NULL;
        }
        atomic_init(&current_thread_arena->offset, 0);
    }

    size_t current_offset = atomic_load(&current_thread_arena->offset);
    while (1) {
        if (current_offset + aligned_size > current_thread_arena->capacity) {
            return NULL; 
        }
        if (atomic_compare_exchange_weak(&current_thread_arena->offset, &current_offset, current_offset + aligned_size)) {
            return (void*)(current_thread_arena->base_ptr + current_offset);
        }
    }
}

VectorChunk* vector_chunk_create(size_t capacity, DataType type) {
    size_t data_size = capacity * sizeof(double); 
    size_t bitmap_size = (capacity + 63) / 64 * sizeof(uint64_t);
    size_t total_size = sizeof(VectorChunk) + data_size + bitmap_size;

    void* mem = aligned_alloc(CACHE_LINE_SIZE, (total_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1));
    if (!mem) return NULL;

    memset(mem, 0, total_size);

    VectorChunk* chunk = (VectorChunk*)mem;
    chunk->capacity = capacity;
    chunk->size = 0;
    chunk->type = type;
    atomic_init(&chunk->version, 1);
    atomic_init(&chunk->ref_count, 1);

    uint8_t* ptr = (uint8_t*)(chunk + 1);
    chunk->data = ptr;
    chunk->null_bitmap = (uint64_t*)(ptr + data_size);

    return chunk;
}

void vector_chunk_destroy(VectorChunk* chunk) {
    if (!chunk) return;
    if (atomic_fetch_sub(&chunk->ref_count, 1) == 1) {
        free(chunk);
    }
}

VectorBatch* vector_batch_create(size_t column_count, size_t capacity) {
    size_t total_size = sizeof(VectorBatch) + (column_count * sizeof(VectorChunk*));
    VectorBatch* batch = (VectorBatch*)aligned_alloc(CACHE_LINE_SIZE, (total_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1));
    if (!batch) return NULL;

    batch->column_count = column_count;
    batch->capacity = capacity;
    batch->columns = (VectorChunk**)(batch + 1);

    for (size_t i = 0; i < column_count; ++i) {
        batch->columns[i] = vector_chunk_create(capacity, TYPE_FLOAT64);
        if (!batch->columns[i]) {
            for (size_t j = 0; j < i; ++j) {
                vector_chunk_destroy(batch->columns[j]);
            }
            free(batch);
            return NULL;
        }
    }
    return batch;
}

void vector_batch_destroy(VectorBatch* batch) {
    if (!batch) return;
    for (size_t i = 0; i < batch->column_count; ++i) {
        vector_chunk_destroy(batch->columns[i]);
    }
    free(batch);
}

ExecutionStatus vector_exec_filter_f64(const VectorChunk* restrict input, 
                                     const double threshold, 
                                     ComparisonOp op, 
                                     SelectionVector* restrict sel) {
    if (!input || !sel) return EXEC_STATUS_ERROR;
    if (input->type != TYPE_FLOAT64) return EXEC_STATUS_TYPE_MISMATCH;

    const double* restrict data = (const double*)input->data;
    const uint64_t* restrict bitmap = input->null_bitmap;
    size_t size = input->size;
    size_t match_count = 0;

    __m512d v_threshold = _mm512_set1_pd(threshold);

    size_t i = 0;
    for (; i + 7 < size; i += 8) {
        uint64_t valid_mask = bitmap ? ~(bitmap[i / 64] >> (i % 64)) : ~0ULL; 
        __m512d v_data = _mm512_loadu_pd(data + i);
        __mmask8 cmp_mask = 0;

        switch (op) {
            case OP_EQ: cmp_mask = _mm512_cmp_pd_mask(v_data, v_threshold, _CMP_EQ_OQ); break;
            case OP_NE: cmp_mask = _mm512_cmp_pd_mask(v_data, v_threshold, _CMP_NEQ_OQ); break;
            case OP_LT: cmp_mask = _mm512_cmp_pd_mask(v_data, v_threshold, _CMP_LT_OS); break;
            case OP_LE: cmp_mask = _mm512_cmp_pd_mask(v_data, v_threshold, _CMP_LE_OS); break;
            case OP_GT: cmp_mask = _mm512_cmp_pd_mask(v_data, v_threshold, _CMP_GT_OS); break;
            case OP_GE: cmp_mask = _mm512_cmp_pd_mask(v_data, v_threshold, _CMP_GE_OS); break;
        }

        uint8_t final_mask = (uint8_t)(cmp_mask) & (uint8_t)(valid_mask);
        
        while (final_mask) {
            unsigned long idx;
            #if defined(_MSC_VER) && !defined(_M_X64)
            unsigned long trailing;
            _BitScanForward(&trailing, final_mask);
            idx = trailing;
            #else
            idx = __builtin_ctz(final_mask);
            #endif
            sel->indices[match_count++] = (uint32_t)(i + idx);
            final_mask &= (final_mask - 1);
        }
    }

    for (; i < size; ++i) {
        if (bitmap && (bitmap[i / 64] & (1ULL << (i % 64)))) continue;
        double val = data[i];
        bool match = false;
        switch (op) {
            case OP_EQ: match = (val == threshold); break;
            case OP_NE: match = (val != threshold); break;
            case OP_LT: match = (val < threshold); break;
            case OP_LE: match = (val <= threshold); break;
            case OP_GT: match = (val > threshold); break;
            case OP_GE: match = (val >= threshold); break;
        }
        if (match) {
            sel->indices[match_count++] = (uint32_t)i;
        }
    }

    sel->size = match_count;
    return EXEC_STATUS_SUCCESS;
}

ExecutionStatus vector_exec_project_add_f64(const VectorChunk* restrict left,
                                            const VectorChunk* restrict right,
                                            VectorChunk* restrict output) {
    if (!left || !right || !output) return EXEC_STATUS_ERROR;
    if (left->size != right->size || left->size > output->capacity) return EXEC_STATUS_BOUNDS_ERROR;
    if (left->type != TYPE_FLOAT64 || right->type != TYPE_FLOAT64 || output->type != TYPE_FLOAT64) 
        return EXEC_STATUS_TYPE_MISMATCH;

    const double* restrict l_data = (const double*)left->data;
    const double* restrict r_data = (const double*)right->data;
    double* restrict o_data = (double*)output->data;
    size_t size = left->size;

    size_t i = 0;
    for (; i + 7 < size; i += 8) {
        __m512d v_l = _mm512_loadu_pd(l_data + i);
        __m512d v_r = _mm512_loadu_pd(r_data + i);
        __m512d v_res = _mm512_add_pd(v_l, v_r);
        _mm512_storeu_pd(o_data + i, v_res);
    }

    for (; i < size; ++i) {
        o_data[i] = l_data[i] + r_data[i];
    }

    output->size = size;
    if (left->null_bitmap || right->null_bitmap) {
        size_t bitmap_words = (size + 63) / 64;
        for (size_t w = 0; w < bitmap_words; ++w) {
            uint64_t l_bits = left->null_bitmap ? left->null_bitmap[w] : 0;
            uint64_t r_bits = right->null_bitmap ? right->null_bitmap[w] : 0;
            output->null_bitmap[w] = l_bits | r_bits;
        }
    }

    atomic_fetch_add(&output->version, 1);
    return EXEC_STATUS_SUCCESS;
}

ExecutionStatus vector_exec_aggregate_sum_f64(const VectorChunk* restrict chunk,
                                              const SelectionVector* restrict sel,
                                              double* restrict result) {
    if (!chunk || !result) return EXEC_STATUS_ERROR;
    if (chunk->type != TYPE_FLOAT64) return EXEC_STATUS_TYPE_MISMATCH;

    const double* restrict data = (const double*)chunk->data;
    const uint64_t* restrict bitmap = chunk->null_bitmap;
    double sum = 0.0;
    __m512d v_sum = _mm512_setzero_pd();

    if (sel) {
        for (size_t i = 0; i < sel->size; ++i) {
            uint32_t idx = sel->indices[i];
            if (bitmap && (bitmap[idx / 64] & (1ULL << (idx % 64)))) continue;
            sum += data[idx];
        }
    } else {
        size_t size = chunk->size;
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            uint64_t valid_mask = bitmap ? ~(bitmap[i / 64] >> (i % 64)) : ~0ULL;
            __m512d v_data = _mm512_maskz_loadu_pd((__mmask8)valid_mask, data + i);
            v_sum = _mm512_add_pd(v_sum, v_data);
        }
        
        double buffer[8];
        _mm512_storeu_pd(buffer, v_sum);
        for (int j = 0; j < 8; ++j) sum += buffer[j];

        for (; i < size; ++i) {
            if (bitmap && (bitmap[i / 64] & (1ULL << (i % 64)))) continue;
            sum += data[i];
        }
    }

    *result = sum;
    return EXEC_STATUS_SUCCESS;
}

SelectionVector* selection_vector_create(size_t capacity) {
    size_t total_size = sizeof(SelectionVector) + (capacity * sizeof(uint32_t));
    SelectionVector* sel = (SelectionVector*)aligned_alloc(CACHE_LINE_SIZE, (total_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1));
    if (!sel) return NULL;
    sel->capacity = capacity;
    sel->size = 0;
    sel->indices = (uint32_t*)(sel + 1);
    return sel;
}

void selection_vector_destroy(SelectionVector* sel) {
    if (!sel) return;
    free(sel);
}