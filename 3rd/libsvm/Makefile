# Minimalistic makefile for libsvm
LIB=libsvm.a
MISC=svm.cpp

LIBFLAGS=-fPIC
CFLAGS=-g -Wall -O2

all: $(LIB)

.cpp.o:
	g++ $(CFLAGS) $(LIBFLAGS) -c $^

$(LIB): $(MISC:.cpp=.o)
	ar cur $@ $^

clean:
	rm -f $(LIB) $(MISC:.cpp=.o)
