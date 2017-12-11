# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -lfftw3

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

# Linker options
LDFLAGS := -fopenmp


BIN = "/usr/local/gcc/6.4.0/bin/gcc"


.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : $(EXEC)

OPENFLAG := -fopenmp 

all : p1

p1: pws_cpu.c
	gcc -o p1 $(OPT) pws_cpu.c -ccbin $(BIN)

.PHONY: clean
clean:
	@ rm -f *.o $(EXEC)

#gcc pws_cpu.c -o pws_cpu -lfftw3 -m64
