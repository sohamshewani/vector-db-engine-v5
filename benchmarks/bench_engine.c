#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define VECTOR_DIM 128
#define WARMUP_ITERATIONS 1000
#define MEASURE_ITERATIONS 50000

typedef struct {
    float data[VECTOR_DIM];
} Vector;

typedef struct {
    uint64_t *latencies;
    size_t count;
    size_t capacity;
} LatencyStats;

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static float dummy_cosine_similarity(const float *a, const float *b, int dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

static int compare_uint64(const void *a, const void *b) {
    uint64_t arg1 = *(const uint64_t *)a;
    uint64_t arg2 = *(const uint64_t *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int main(void) {
    srand(42);

    Vector *query = malloc(sizeof(Vector));
    Vector *database = malloc(sizeof(Vector) * 1000);

    for (int i = 0; i < VECTOR_DIM; i++) {
        query->data[i] = ((float)rand() / (float)RAND_MAX);
    }
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < VECTOR_DIM; i++) {
            database[j].data[i] = ((float)rand() / (float)RAND_MAX);
        }
    }

    uint64_t *latencies = malloc(sizeof(uint64_t) * MEASURE_ITERATIONS);
    if (!latencies) {
        perror("Failed to allocate memory for latencies");
        return 1;
    }

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        volatile float res = 0.0f;
        for (int j = 0; j < 1000; j++) {
            res += dummy_cosine_similarity(query->data, database[j].data, VECTOR_DIM);
        }
        (void)res;
    }

    uint64_t total_start = get_time_ns();

    for (size_t i = 0; i < MEASURE_ITERATIONS; i++) {
        uint64_t start = get_time_ns();
        
        volatile float best_score = -1.0f;
        for (int j = 0; j < 1000; j++) {
            float score = dummy_cosine_similarity(query->data, database[j].data, VECTOR_DIM);
            if (score > best_score) {
                best_score = score;
            }
        }
        
        uint64_t end = get_time_ns();
        latencies[i] = end - start;
    }

    uint64_t total_end = get_time_ns();
    uint64_t total_duration_ns = total_end - total_start;

    qsort(latencies, MEASURE_ITERATIONS, sizeof(uint64_t), compare_uint64);

    uint64_t p50 = latencies[(size_t)(MEASURE_ITERATIONS * 0.50)];
    uint64_t p90 = latencies[(size_t)(MEASURE_ITERATIONS * 0.90)];
    uint64_t p99 = latencies[(size_t)(MEASURE_ITERATIONS * 0.99)];
    uint64_t p999 = latencies[(size_t)(MEASURE_ITERATIONS * 0.999)];

    double total_seconds = (double)total_duration_ns / 1e9;
    double ops_per_sec = (double)MEASURE_ITERATIONS / total_seconds;

    printf("\n");
    printf("+------------------------------------------------+\n");
    printf("|       vector-db-engine Microbenchmark          |\n");
    printf("+------------------------+-----------------------+\n");
    printf("| Metric                 | Value                 |\n");
    printf("+------------------------+-----------------------+\n");
    printf("| Total Operations       | %-21u |\n", (unsigned int)MEASURE_ITERATIONS);
    printf("| Vector Dimensions      | %-21d |\n", VECTOR_DIM);
    printf("| Database Size          | %-21d |\n", 1000);
    printf("| Throughput             | %15.2f ops/s |\n", ops_per_sec);
    printf("| Latency P50            | %15.2f us  |\n", (double)p50 / 1000.0);
    printf("| Latency P90            | %15.2f us  |\n", (double)p90 / 1000.0);
    printf("| Latency P99            | %15.2f us  |\n", (double)p99 / 1000.0);
    printf("| Latency P99.9          | %15.2f us  |\n", (double)p999 / 1000.0);
    printf("+------------------------+-----------------------+\n");
    printf("\n");

    free(query);
    free(database);
    free(latencies);

    return 0;
}