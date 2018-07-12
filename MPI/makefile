TARGET=a.out

CC=mpicc
CFLAGS=-O3 -mp -acc -Minfo=accel

SRC= laplace_mpi.c
OBJ=$(SRC:.cpp=.o)

.SUFFIXES: .c .cc .cpp .h. .o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJ)

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJ)
