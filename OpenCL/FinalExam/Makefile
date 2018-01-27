UTILS =
CC = cc
CFLAGS = -Wall -Wno-comment -Wno-deprecated-declarations -O2 -std=gnu99 -DIMAGE_W=2048 -DIMAGE_H=2048 -DFILTER_W=13
TARGET = convolveCL

# Linux
LIBS = -lOpenCL -lm


all: ${TARGET}

${TARGET}: ${TARGET}.c ${UTILS}
	${CC} ${CFLAGS} ${LIBS} ${UTILS} ${TARGET}.c -o ${TARGET}

clean:
	rm -f ${TARGET}

