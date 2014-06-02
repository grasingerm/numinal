CC=g++
CFLAGS=-Wall -Wextra -pedantic -pedantic-errors -g -std=c++11 -O3

all: pi inv_e gauss

pi: pi.cpp
	$(CC) $(CFLAGS) pi.cpp -o pi
	
inv_e: inv_e.cpp
	$(CC) $(CFLAGS) inv_e.cpp -o inv_e
	
gauss: gauss.cpp
	$(CC) $(CFLAGS) gauss.cpp -o gauss -larmadillo
	
clean:
	rm *.o
