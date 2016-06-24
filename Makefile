CC 	= gcc
MPICC	= mpicc
LD	= gcc
MPILD	= mpicc

CFLAGS  	= -Wall -O3 -fomit-frame-pointer -funroll-loops
MPICCFLAGS 	= -Wall -O3 -fomit-frame-pointer -funroll-loops
LFLAGS  =

#MPICCFLAGS += -DPRINT_DATA
#CFLAGS += -DPRINT_DATA

PROGS    = serial_freccia parallel_freccia 

.PHONY:	clean

all: ${PROGS}

serial_freccia.o: serial_freccia.c
	        $(CC) $(CFLAGS) -c -o $*.o $<

serial_freccia: serial_freccia.o
	        $(LD) $(LFLAGS) -o $@ $^
		
parallel_freccia.o: parallel_freccia.c
	        $(MPICC) $(MPICCFLAGS) -c -o $*.o $<

parallel_freccia: parallel_freccia.o
	        $(MPILD) $(LFLAGS) -o $@ $^


clean:
	rm -f *.o $(PROGS)

