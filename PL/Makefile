CC=g++
CFLAGS=-Wall -Wextra -pedantic -pedantic-errors -g -std=c++11 -O3

all: euclid polynom sqrt_a

euclid: euclid.cpp
	$(CC) $(CFLAGS) euclid.cpp -o euclid
	
polynom: polynom.cpp
	$(CC) $(CFLAGS) polynom.cpp -o polynom
	
sqrt_a: sqrt_a.cpp
	$(CC) $(CFLAGS) sqrt_a.cpp -o sqrt_a
	
clean:
	rm *.o
