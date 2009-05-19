# Minimalistic makefile for libsvm
LIB=libsvm.a
MISC=svm.cpp

LIBFLAGS=-fPIC
CFLAGS=-Wall -O2

TDIR=../../build/libsvm

all: prep $(LIB)

prep: prep-stamp
prep-stamp:
	mkdir -p $(TDIR)
	touch $@

.cpp.o:
	g++ $(CFLAGS) $(LIBFLAGS) -c $^ -o $(TDIR)/$(^:.cpp=.o)

$(LIB): $(MISC:.cpp=.o)
	ar cur $(TDIR)/$@ $(TDIR)/$^

clean:
	rm -rf $(TDIR)
	rm -f *-stamp
