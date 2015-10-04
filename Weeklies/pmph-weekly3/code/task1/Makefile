CC=gcc
CFLAGS=-c -Wall -fopenmp
LDFLAGS=

BINDIR=./bin

EXECUTABLE=$(BINDIR)/matrix

SOURCES=matrix.cu
OBJECTS=$(SOURCES:.c=.o)

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cu.o:
	$(CC) $(CFLAGS) $< -o $@

run:
	$(EXECUTABLE)

clean:
	rm $(EXECUTABLE) $(OBJECTS)
