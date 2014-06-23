CC=g++
CFLAGS=-Wall -Wextra -pedantic -pedantic-errors -g -std=c++11 -O3

all: pi inv_e gauss lu ldl finite_difference

pi: pi.cpp
	$(CC) $(CFLAGS) pi.cpp -o pi
	
inv_e: inv_e.cpp
	$(CC) $(CFLAGS) inv_e.cpp -o inv_e
	
gauss: gauss.cpp
	$(CC) $(CFLAGS) gauss.cpp -o gauss -larmadillo
	
lu: lu.cpp
	$(CC) $(CFLAGS) lu.cpp -o lu -larmadillo
	
ldl: ldl.cpp
	$(CC) $(CFLAGS) ldl.cpp -o ldl -larmadillo
	
finite_difference: finite_difference.cpp assertion_helper.h
	$(CC) $(CFLAGS) finite_difference.cpp -o finite_difference -larmadillo
	
clean:
	rm -f *.o pi inv_e gauss lu ldl finite_difference
