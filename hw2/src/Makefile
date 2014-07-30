CC=g++
CFLAGS=-Wall -Wextra -pedantic -pedantic-errors -g -std=c++11 -O3

all: s91

s91: s91.cpp
	$(CC) $(CFLAGS) s91.cpp -o s91 -larmadillo
	
clean:
	rm -f *.o s91
