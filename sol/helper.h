#ifndef ACCPOP_HELPER_H
#define ACCPOP_HELPER_H
#include <stdio.h>
float geoDistance(float lat1, float lon1, float lat2, float lon2);

#define DEGREE_TO_RADIANS		0.01745329252f

void sampleFileIO(float kmRange, const char* fileIn, const char* fileOut);

#define DIE(assertion, call_description)                    \
do {                                                        \
    if (assertion) {                                        \
            fprintf(stderr, "(%d): ",                       \
                            __LINE__);                      \
            perror(call_description);                       \
            exit(EXIT_FAILURE);                             \
    }                                                       \
} while(0);

#endif //ACCPOP_HELPER_H
