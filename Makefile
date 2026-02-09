# ═══════════════════════════════════════════════════════════════════════════════
# HEXSTATE ENGINE — 6-State Quantum Processor
# ═══════════════════════════════════════════════════════════════════════════════
# Build: make
# Clean: make clean
# Test:  make test

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDFLAGS = -lm

TARGET  = hexstate_engine
STRESS  = stress_test
SRCS    = main.c hexstate_engine.c bigint.c
OBJS    = $(SRCS:.c=.o)

.PHONY: all clean test stress bell crystal lib

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(STRESS): stress_test.o bigint.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bell_test: bell_test.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

time_crystal: time_crystal_test.o hexstate_engine.o bigint.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Shared library for external drivers (Python, etc.)
LIBNAME = libhexstate.so

$(LIBNAME): hexstate_engine.c bigint.c hexstate_engine.h bigint.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ hexstate_engine.c bigint.c $(LDFLAGS)

lib: $(LIBNAME)
	@echo "Built $(LIBNAME) — load from Python: ctypes.CDLL('./$(LIBNAME)')"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Dependencies
main.o: main.c hexstate_engine.h bigint.h
hexstate_engine.o: hexstate_engine.c hexstate_engine.h bigint.h
bigint.o: bigint.c bigint.h
stress_test.o: stress_test.c hexstate_engine.h bigint.h
bell_test.o: bell_test.c hexstate_engine.h
time_crystal_test.o: time_crystal_test.c hexstate_engine.h bigint.h

test: $(TARGET)
	./$(TARGET) --self-test

stress: $(STRESS)
	./$(STRESS)

bell: bell_test
	./bell_test

crystal: time_crystal
	./time_crystal

clean:
	rm -f $(OBJS) stress_test.o bell_test.o time_crystal_test.o $(TARGET) $(STRESS) bell_test time_crystal $(LIBNAME)
