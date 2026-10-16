#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <errno.h>

/* ================================================================= *
 * VECTOR-DB-ENGINE SOURCE (Embedded for self-contained execution)   *
 * ================================================================= */

#define VDB_OK 0
#define VDB_ERR_NOMEM -1
#define VDB_ERR_INVALID -2
#define VDB_ERR_NOTFOUND -3
#define VDB_ERR_FULL -4
#define VDB_ERR_CORRUPT -5

typedef uint64_t vdb_id_t;

typedef struct {
    vdb_id_t id;
    uint32_t dim;
    float *data;
} vdb_vector_t;

typedef struct {
    uint32_t dim;
    size_t capacity;
    size_t count;
    vdb_vector_t *vectors;
    pthread_rwlock_t rwlock;
    bool is_corrupted;
} vdb_collection_t;

typedef struct {
    vdb_id_t id;
    float distance;
} vdb_query_result_t;

static inline float vdb_cosine_similarity(const float *a, const float *b, uint32_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

vdb_collection_t* vdb_collection_create(uint32_t dim, size_t capacity) {
    if (dim == 0 || capacity == 0) return NULL;
    vdb_collection_t *col = malloc(sizeof(vdb_collection_t));
    if (!col) return NULL;
    
    col->dim = dim;
    col->capacity = capacity;
    col->count = 0;
    col->is_corrupted = false;
    col->vectors = malloc(capacity * sizeof(vdb_vector_t));
    if (!col->vectors) {
        free(col);
        return NULL;
    }
    if (pthread_rwlock_init(&col->rwlock, NULL) != 0) {
        free(col->vectors);
        free(col);
        return NULL;
    }
    return col;
}

void vdb_collection_free(vdb_collection_t *col) {
    if (!col) return;
    pthread_rwlock_wrlock(&col->rwlock);
    for (size_t i = 0; i < col->count; ++i) {
        free(col->vectors[i].data);
    }
    free(col->vectors);
    pthread_rwlock_unlock(&col->rwlock);
    pthread_rwlock_destroy(&col->rwlock);
    free(col);
}

int vdb_insert(vdb_collection_t *col, vdb_id_t id, const float *data) {
    if (!col || !data) return VDB_ERR_INVALID;
    
    pthread_rwlock_wrlock(&col->rwlock);
    if (col->is_corrupted) {
        pthread_rwlock_unlock(&col->rwlock);
        return VDB_ERR_CORRUPT;
    }
    
    for (size_t i = 0; i < col->count; ++i) {
        if (col->vectors[i].id == id) {
            pthread_rwlock_unlock(&col->rwlock);
            return VDB_ERR_INVALID;
        }
    }
    
    if (col->count >= col->capacity) {
        size_t new_cap = col->capacity * 2;
        vdb_vector_t *new_vecs = realloc(col->vectors, new_cap * sizeof(vdb_vector_t));
        if (!new_vecs) {
            pthread_rwlock_unlock(&col->rwlock);
            return VDB_ERR_NOMEM;
        }
        col->vectors = new_vecs;
        col->capacity = new_cap;
    }
    
    float *vec_data = malloc(col->dim * sizeof(float));
    if (!vec_data) {
        pthread_rwlock_unlock(&col->rwlock);
        return VDB_ERR_NOMEM;
    }
    memcpy(vec_data, data, col->dim * sizeof(float));
    
    col->vectors[col->count].id = id;
    col->vectors[col->count].dim = col->dim;
    col->vectors[col->count].data = vec_data;
    col->count++;
    
    pthread_rwlock_unlock(&col->rwlock);
    return VDB_OK;
}

int vdb_get(vdb_collection_t *col, vdb_id_t id, float *out_data) {
    if (!col || !out_data) return VDB_ERR_INVALID;
    
    pthread_rwlock_rdlock(&col->rwlock);
    if (col->is_corrupted) {
        pthread_rwlock_unlock(&col->rwlock);
        return VDB_ERR_CORRUPT;
    }
    
    for (size_t i = 0; i < col->count; ++i) {
        if (col->vectors[i].id == id) {
            memcpy(out_data, col->vectors[i].data, col->dim * sizeof(float));
            pthread_rwlock_unlock(&col->rwlock);
            return VDB_OK;
        }
    }
    
    pthread_rwlock_unlock(&col->rwlock);
    return VDB_ERR_NOTFOUND;
}

int vdb_delete(vdb_collection_t *col, vdb_id_t id) {
    if (!col) return VDB_ERR_INVALID;
    
    pthread_rwlock_wrlock(&col->rwlock);
    if (col->is_corrupted) {
        pthread_rwlock_unlock(&col->rwlock);
        return VDB_ERR_CORRUPT;
    }
    
    for (size_t i = 0; i < col->count; ++i) {
        if (col->vectors[i].id == id) {
            free(col->vectors[i].data);
            col->vectors[i] = col->vectors[col->count - 1];
            col->count--;
            pthread_rwlock_unlock(&col->rwlock);
            return VDB_OK;
        }
    }
    
    pthread_rwlock_unlock(&col->rwlock);
    return VDB_ERR_NOTFOUND;
}

int vdb_search(vdb_collection_t *col, const float *query, size_t k, vdb_query_result_t *results, size_t *out_k) {
    if (!col || !query || !results || !out_k || k == 0) return VDB_ERR_INVALID;
    
    pthread_rwlock_rdlock(&col->rwlock);
    if (col->is_corrupted) {
        pthread_rwlock_unlock(&col->rwlock);
        return VDB_ERR_CORRUPT;
    }
    
    if (col->count == 0) {
        *out_k = 0;
        pthread_rwlock_unlock(&col->rwlock);
        return VDB_OK;
    }
    
    size_t n = col->count < k ? col->count : k;
    
    for (size_t i = 0; i < n; ++i) {
        results[i].id = col->vectors[i].id;
        results[i].distance = vdb_cosine_similarity(query, col->vectors[i].data, col->dim);
    }
    
    for (size_t i = 0; i < col->count; ++i) {
        float sim = vdb_cosine_similarity(query, col->vectors[i].data, col->dim);
        for (size_t j = 0; j < n; ++j) {
            if (sim > results[j].distance) {
                for (size_t m = n - 1; m > j; --m) {
                    results[m] = results[m - 1];
                }
                results[j].id = col->vectors[i].id;
                results[j].distance = sim;
                break;
            }
        }
    }
    
    *out_k = n;
    pthread_rwlock_unlock(&col->rwlock);
    return VDB_OK;
}

void vdb_inject_corruption(vdb_collection_t *col) {
    if (!col) return;
    pthread_rwlock_wrlock(&col->rwlock);
    col->is_corrupted = true;
    pthread_rwlock_unlock(&col->rwlock);
}

void vdb_recover(vdb_collection_t *col) {
    if (!col) return;
    pthread_rwlock_wrlock(&col->rwlock);
    for (size_t i = 0; i < col->count; ++i) {
        free(col->vectors[i].data);
    }
    col->count = 0;
    col->is_corrupted = false;
    pthread_rwlock_unlock(&col->rwlock);
}


/* ================================================================= *
 * TEST SUITE IMPLEMENTATION                                       *
 * ================================================================= */

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "\033[31m[FAIL]\033[0m %s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    printf("[RUN ] %s...\n", #test_func); \
    test_func(); \
    printf("\033[32m[PASS]\033[0m %s\n", #test_func); \
} while(0)

static void test_unit_boundaries(void) {
    // Zero Dimension
    vdb_collection_t *col = vdb_collection_create(0, 10);
    TEST_ASSERT(col == NULL);

    // Zero Capacity
    col = vdb_collection_create(128, 0);
    TEST_ASSERT(col == NULL);

    col = vdb_collection_create(4, 2);
    TEST_ASSERT(col != NULL);

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[4] = {0.0f, 0.0f, 1.0f, 0.0f};

    // Insert within capacity
    TEST_ASSERT(vdb_insert(col, 1, v1) == VDB_OK);
    TEST_ASSERT(vdb_insert(col, 2, v2) == VDB_OK);

    // Duplicate ID edge case
    TEST_ASSERT(vdb_insert(col, 1, v3) == VDB_ERR_INVALID);

    // Trigger reallocation (capacity 2 -> 4)
    TEST_ASSERT(vdb_insert(col, 3, v3) == VDB_OK);
    TEST_ASSERT(col->capacity == 4);
    TEST_ASSERT(col->count == 3);

    // Get non-existent
    float out[4];