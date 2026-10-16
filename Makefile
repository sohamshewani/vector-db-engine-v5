CC = gcc
CFLAGS = -std=c11 -O3 -Wall -Wextra -Wpedantic -march=native
LDFLAGS = -lm -lpthread

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
TEST_DIR = tests
BENCH_DIR = benchmarks

TARGET = $(BUILD_DIR)/vector-db-engine
TEST_TARGET = $(BUILD_DIR)/test-runner
BENCH_TARGET = $(BUILD_DIR)/benchmark-runner

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
BENCH_SRCS = $(wildcard $(BENCH_DIR)/*.c)

.PHONY: all build test benchmark clean directories

all: directories build

directories:
	@mkdir -p $(BUILD_DIR)

build: directories $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -I$(INC_DIR) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

test: directories
	@if [ -d "$(TEST_DIR)" ] && [ -n "$$(ls -A $(TEST_DIR) 2>/dev/null)" ]; then \
		$(CC) $(CFLAGS) -I$(INC_DIR) $(TEST_SRCS) $(filter-out $(BUILD_DIR)/main.o, $(OBJS)) -o $(TEST_TARGET) $(LDFLAGS); \
		$(TEST_TARGET); \
	else \
		echo "No tests found in $(TEST_DIR). Skipping."; \
	fi

benchmark: directories
	@if [ -d "$(BENCH_DIR)" ] && [ -n "$$(ls -A $(BENCH_DIR) 2>/dev/null)" ]; then \
		$(CC) $(CFLAGS) -I$(INC_DIR) $(BENCH_SRCS) $(filter-out $(BUILD_DIR)/main.o, $(OBJS)) -o $(BENCH_TARGET) $(LDFLAGS); \
		$(BENCH_TARGET); \
	else \
		echo "No benchmarks found in $(BENCH_DIR). Skipping."; \
	fi

clean:
	@rm -rf $(BUILD_DIR)