#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

#define CACHE_LINE_SIZE 64
#define COMPRESSION_SUCCESS 0
#define COMPRESSION_ERR_OOM -1
#define COMPRESSION_ERR_INVALID_ARG -2
#define COMPRESSION_ERR_OVERFLOW -3

typedef struct {
    uint8_t* data;
    size_t capacity;
    size_t size;
} __attribute__((aligned(CACHE_LINE_SIZE))) CompressionArena;

static inline void* arena_alloc(CompressionArena* arena, size_t size) {
    size_t aligned_size = (size + (CACHE_LINE_SIZE - 1)) & ~(CACHE_LINE_SIZE - 1);
    if (arena->size + aligned_size > arena->capacity) {
        return NULL;
    }
    void* ptr = arena->data + arena->size;
    arena->size += aligned_size;
    return ptr;
}

static inline uint8_t get_bit_width_u64(uint64_t val) {
    if (val == 0) return 1;
    return (uint8_t)(64 - __builtin_clzll(val));
}

static inline uint8_t get_max_bit_width(const uint64_t* restrict values, size_t count) {
    uint64_t max_val = 0;
    size_t i = 0;
    
    for (; i + 3 < count; i += 4) {
        max_val |= values[i] | values[i+1] | values[i+2] | values[i+3];
    }
    for (; i < count; ++i) {
        max_val |= values[i];
    }
    return get_bit_width_u64(max_val);
}

int compression_arena_init(CompressionArena* arena, size_t capacity) {
    if (!arena || capacity == 0) return COMPRESSION_ERR_INVALID_ARG;
    size_t aligned_cap = (capacity + (CACHE_LINE_SIZE - 1)) & ~(CACHE_LINE_SIZE - 1);
    
    int ret = posix_memalign((void**)&arena->data, CACHE_LINE_SIZE, aligned_cap);
    if (ret != 0 || !arena->data) {
        return COMPRESSION_ERR_OOM;
    }
    arena->capacity = aligned_cap;
    arena->size = 0;
    return COMPRESSION_SUCCESS;
}

void compression_arena_free(CompressionArena* arena) {
    if (arena && arena->data) {
        free(arena->data);
        arena->data = NULL;
        arena->capacity = 0;
        arena->size = 0;
    }
}

int rle_encode_u64(const uint64_t* restrict input, size_t count, 
                   CompressionArena* arena, 
                   uint8_t** out_runs, uint8_t** out_values, size_t* out_encoded_count) {
    if (!input || !arena || !out_runs || !out_values || !out_encoded_count) {
        return COMPRESSION_ERR_INVALID_ARG;
    }
    if (count == 0) {
        *out_runs = NULL;
        *out_values = NULL;
        *out_encoded_count = 0;
        return COMPRESSION_SUCCESS;
    }

    size_t max_runs_bytes = count * sizeof(uint32_t);
    size_t max_vals_bytes = count * sizeof(uint64_t);

    uint32_t* runs_buf = (uint32_t*)arena_alloc(arena, max_runs_bytes);
    uint64_t* vals_buf = (uint64_t*)arena_alloc(arena, max_vals_bytes);

    if (!runs_buf || !vals_buf) {
        return COMPRESSION_ERR_OOM;
    }

    size_t run_idx = 0;
    uint64_t current_val = input[0];
    uint32_t current_run = 1;

    for (size_t i = 1; i < count; ++i) {
        if (input[i] == current_val && current_run < UINT32_MAX) {
            current_run++;
        } else {
            runs_buf[run_idx] = current_run;
            vals_buf[run_idx] = current_val;
            run_idx++;
            current_val = input[i];
            current_run = 1;
        }
    }
    runs_buf[run_idx] = current_run;
    vals_buf[run_idx] = current_val;
    run_idx++;

    *out_runs = (uint8_t*)runs_buf;
    *out_values = (uint8_t*)vals_buf;
    *out_encoded_count = run_idx;

    return COMPRESSION_SUCCESS;
}

int rle_decode_u64(const uint8_t* restrict runs, const uint8_t* restrict values, size_t encoded_count,
                   uint64_t* restrict output, size_t output_capacity, size_t* restrict out_decoded_count) {
    if (!runs || !values || !output || !out_decoded_count) {
        return COMPRESSION_ERR_INVALID_ARG;
    }

    const uint32_t* r = (const uint32_t*)runs;
    const uint64_t* v = (const uint64_t*)values;
    size_t total_decoded = 0;

    for (size_t i = 0; i < encoded_count; ++i) {
        uint32_t run_len = r[i];
        uint64_t val = v[i];

        if (total_decoded + run_len > output_capacity) {
            return COMPRESSION_ERR_OVERFLOW;
        }

        size_t j = 0;
        for (; j + 7 < run_len; j += 8) {
            output[total_decoded + j + 0] = val;
            output[total_decoded + j + 1] = val;
            output[total_decoded + j + 2] = val;
            output[total_decoded + j + 3] = val;
            output[total_decoded + j + 4] = val;
            output[total_decoded + j + 5] = val;
            output[total_decoded + j + 6] = val;
            output[total_decoded + j + 7] = val;
        }
        for (; j < run_len; ++j) {
            output[total_decoded + j] = val;
        }
        total_decoded += run_len;
    }

    *out_decoded_count = total_decoded;
    return COMPRESSION_SUCCESS;
}

int bitpack_encode_u64(const uint64_t* restrict input, size_t count,
                       CompressionArena* arena,
                       uint8_t** out_packed, uint8_t* out_bit_width) {
    if (!input || !arena || !out_packed || !out_bit_width || count == 0) {
        return COMPRESSION_ERR_INVALID_ARG;
    }

    uint8_t bit_width = get_max_bit_width(input, count);
    *out_bit_width = bit_width;

    size_t total_bits = (size_t)count * bit_width;
    size_t packed_bytes = (total_bits + 7) / 8;
    size_t aligned_packed_bytes = (packed_bytes + (CACHE_LINE_SIZE - 1)) & ~(CACHE_LINE_SIZE - 1);

    uint8_t* packed_buf = (uint8_t*)arena_alloc(arena, aligned_packed_bytes);
    if (!packed_buf) {
        return COMPRESSION_ERR_OOM;
    }
    memset(packed_buf, 0, aligned_packed_bytes);

    if (bit_width == 64) {
        memcpy(packed_buf, input, count * sizeof(uint64_t));
        *out_packed = packed_buf;
        return COMPRESSION_SUCCESS;
    }

    size_t bit_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        uint64_t val = input[i];
        size_t byte_idx = bit_offset >> 3;
        size_t bit_rem = bit_offset & 7;

        uint64_t current_bytes;
        memcpy(&current_bytes, packed_buf + byte_idx, sizeof(uint64_t));
        
        current_bytes |= (val << bit_rem);
        
        memcpy(packed_buf + byte_idx, &current_bytes, sizeof(uint64_t));
        bit_offset += bit_width;
    }

    *out_packed = packed_buf;
    return COMPRESSION_SUCCESS;
}

int bitpack_decode_u64(const uint8_t* restrict packed, uint8_t bit_width, size_t count,
                       uint64_t* restrict output, size_t output_capacity) {
    if (!packed || !output || count == 0) {
        return COMPRESSION_ERR_INVALID_ARG;
    }
    if (count > output_capacity) {
        return COMPRESSION_ERR_OVERFLOW;
    }
    if (bit_width == 0 || bit_width > 64) {
        return COMPRESSION_ERR_INVALID_ARG;
    }

    if (bit_width == 64) {
        memcpy(output, packed, count * sizeof(uint64_t));
        return COMPRESSION_SUCCESS;
    }

    uint64_t mask = (bit_width == 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << bit_width) - 1ULL);
    size_t bit_offset = 0;

    for (size_t i = 0; i < count; ++i) {
        size_t byte_idx = bit_offset >> 3;
        size_t bit_rem = bit_offset & 7;

        uint64_t current_bytes;
        memcpy(&current_bytes, packed + byte_idx, sizeof(uint64_t));

        output[i] = (current_bytes >> bit_rem) & mask;
        bit_offset += bit_width;
    }

    return COMPRESSION_SUCCESS;
}

int dictionary_encode_strings(const char** restrict strings, size_t count,
                              CompressionArena* arena,
                              uint8_t** out_ids, uint8_t* out_id_width,
                              char*** out_dict, size_t* out_dict_size) {
    if (!strings || !arena || !out_ids || !out_id_width || !out_dict || !out_dict_size) {
        return COMPRESSION_ERR_INVALID_ARG;
    }
    if (count == 0) {
        *out_ids = NULL;
        *out_id_width = 0;
        *out_dict = NULL;
        *out_dict_size = 0;
        return COMPRESSION_SUCCESS;
    }

    size_t max_dict_entries = count;
    char** dict_strings = (char**)arena_alloc(arena, max_dict_entries * sizeof(char*));
    size_t* dict_counts = (size_t*)arena_alloc(arena, max_dict_entries * sizeof(size_t));
    uint32_t* raw_ids = (uint32_t*)arena_alloc(arena, count * sizeof(uint32_t));

    if (!dict_strings || !dict_counts || !raw_ids) {
        return COMPRESSION_ERR_OOM;
    }

    size_t unique_count = 0;

    for (size_t i = 0; i < count; ++i) {
        const char* s = strings[i];
        size_t found_id = (size_t)-1;

        for (size_t j = 0; j < unique_count; ++j) {
            if (strcmp(dict_strings[j], s) == 0) {
                found_id = j;
                break;
            }
        }

        if (found_id == (size_t)-1) {
            size_t len = strlen(s) + 1;
            char* dest = (char*)arena_alloc(arena, len);
            if (!dest) return COMPRESSION_ERR_OOM;
            memcpy(dest, s, len);

            dict_strings[unique_count] = dest;
            raw_ids[i] = (uint32_t)unique_count;
            unique_count++;
        } else {
            raw_ids[i] = (uint32_t)found_id;
        }
    }

    uint8_t id_width = 4;
    if (unique_count <= 256) {
        id_width = 1;
    } else if (unique_count <= 65536) {
        id_width = 2;
    } else {
        id_width = 4;
    }

    size_t ids_bytes = count * id_width;
    size_t aligned_ids_bytes = (ids_bytes + (CACHE_LINE_SIZE - 1)) & ~(CACHE_LINE_SIZE - 1);
    uint8_t* ids_buf = (uint8_t*)arena_alloc(arena, aligned_ids_bytes);
    if (!ids_buf) return COMPRESSION_ERR_OOM;
    memset(ids_buf, 0, aligned_ids_bytes);

    if (id_width == 1) {
        uint8_t* p = ids_buf;
        for (size_t i = 0; i < count; ++i) {
            p[i] = (uint8_t)raw_ids[i];
        }
    } else if (id_width == 2) {
        uint16_t* p = (uint16_t*)ids_buf;
        for (size_t i = 0; i < count; ++i) {
            p[i] = (uint16_t)raw_ids[i];
        }
    } else {
        uint32_t* p = (uint32_t*)ids_buf;
        memcpy(p, raw_ids, count * sizeof(uint32_t));
    }

    *out_ids = ids_buf;
    *out_id_width = id_width;
    *out_dict = dict_strings;
    *out_dict_size = unique_count;

    return COMPRESSION_SUCCESS;
}

int dictionary_decode_strings(const uint8_t* restrict ids, uint8_t id_width, size_t count,
                              char** restrict dict, size_t dict_size,
                              char** restrict output) {
    if (!ids || !dict || !output || count == 0) {
        return COMPRESSION_ERR_INVALID_ARG;
    }
    if (id_width != 1 && id_width != 2 && id_width != 4) {
        return COMPRESSION_ERR_INVALID_ARG;
    }

    for (size_t i = 0; i < count; ++i) {
        size_t id = 0;
        if (id_width == 1) {
            id = ids[i];
        } else if (id_width == 2) {
            id = ((const uint16_t*)ids)[i];
        } else {
            id = ((const uint32_t*)ids)[i];
        }

        if (id >= dict_size) {
            return COMPRESSION_ERR_OVERFLOW;
        }
        output[i] = dict[id];
    }

    return COMPRESSION_SUCCESS;
}