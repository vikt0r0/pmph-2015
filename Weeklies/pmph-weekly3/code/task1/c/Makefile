CC=gcc
CFLAGS=-c -Wall
LDFLAGS=

BINDIR=./bin

EXECUTABLE=$(BINDIR)/matrix_transpose_cu_naive

SOURCES=matrix_transpose_seq.cu
OBJECTS=$(SOURCES:.cu=.o)

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

run:
	$(EXECUTABLE)

clean:
	rm $(EXECUTABLE) $(OBJECTS)
