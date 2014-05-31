CC=g++
CFLAGS=-Wall -Wextra -pedantic -pedantic-errors -g -std=c++11 -O3

all: euclid polynom

euclid: euclid.cpp
	$(CC) $(CFLAGS) euclid.cpp -o euclid
	
polynom: polynom.cpp
	$(CC) $(CFLAGS) polynom.cpp -o polynom
	
clean:
	rm *.o
