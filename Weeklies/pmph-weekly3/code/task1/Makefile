CC=gcc
CFLAGS=-c -Wall
LDFLAGS=

BINDIR=./bin

EXECUTABLE=$(BINDIR)/matrix_transpose_seq

SOURCES=matrix_transpose_seq.c
OBJECTS=$(SOURCES:.c=.o)

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

run:
	$(EXECUTABLE)

clean:
	rm $(EXECUTABLE) $(OBJECTS)
