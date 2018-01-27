UTILS =
CC = cc
CFLAGS = -Wall -Wno-comment -Wno-deprecated-declarations -O2 -std=gnu99

TARGET = convolveCL

# Linux
LIBS = -lOpenCL -lm


all: ${TARGET}

${TARGET}: ${TARGET}.c ${UTILS}
	${CC} ${CFLAGS} ${LIBS} ${UTILS} ${TARGET}.c -o ${TARGET}

clean:
	rm -f ${TARGET}

