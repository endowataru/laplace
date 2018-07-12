TARGET=a.out

CC=mpicc
CPPFLAGS=-O3 -mp -acc -Minfo=accel

SRC= laplace_mpi.c
OBJ=$(SRC:.cpp=.o)

.SUFFIXES: .c .cc .cpp .h. .o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CPPFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJ)

.c.o:
	$(CC) $(CPPFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJ)
