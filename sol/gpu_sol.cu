#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <math.h>

#include "helper.h"
#define BLOCK_SIZE 256

unsigned int fill_data(std::vector<float>& longit, std::vector <float>& lat, std::vector<unsigned int>& pop, char *input_filename) {
        std::ifstream ifs(input_filename);

        std::string aux_string;
        float aux_longit;
        float aux_lat;
        unsigned int aux_pop;

        unsigned int no_lines = 0;

        while (ifs >> aux_string >> aux_longit >> aux_lat >> aux_pop) {
                //printf("shoul enter here\n");
                longit.push_back(aux_longit);
                lat.push_back(aux_lat);
                pop.push_back(aux_pop);
                no_lines++;
        }

        ifs.close();
        return no_lines;
}


__global__ void master(unsigned int *results, float *longits, float *lats, unsigned int *pops, unsigned int file_size, float km_range) {
        unsigned int idx =  blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int my_file_size = file_size;
        if (idx >= file_size) return;

        float my_degree_to_radians = DEGREE_TO_RADIANS;
        float ninety_degrees = 90.f;
        float sin_2_cos_2 = 1.0f;

        float sin_phi1 = sin((ninety_degrees - lats[idx]) * my_degree_to_radians);
        float cos_phi1 = sqrt(sin_2_cos_2 - sin_phi1 * sin_phi1);

        float theta1 = longits[idx] * my_degree_to_radians;

        float my_km_range = km_range;
        unsigned int my_pop = pops[idx];

        float constant_dec = ninety_degrees * my_degree_to_radians;

        for (unsigned int i = idx + 1; i < my_file_size; i++) {

                float sin_phi2 = sin(constant_dec - lats[i] * my_degree_to_radians);
                float cos_phi2 = sqrt(sin_2_cos_2 - sin_phi2 * sin_phi2);

                float theta2 = longits[i] * my_degree_to_radians;

                float cs = sin_phi1 * sin_phi2 * cos(theta1 - theta2) + cos_phi1 * cos_phi2;
                if (cs > 1) {
                        cs = 1;
                } else if (cs < -1) {
                        cs = -1;
                }

                if (6371.f * acos(cs) <= my_km_range) {
                        atomicAdd(&results[idx], pops[i]);
                        atomicAdd(&results[i], my_pop);
                }

        }
}

void writeResults(char *output_file_name, unsigned int *results, unsigned int size) {
        std::ofstream ofs(output_file_name);

        for (int i = 0; i < size; i++) {
               ofs << results[i] << "\n";
        }

        ofs.close();
}

int main(int argc, char *argv[]) {
        if (argc == 1) {
                std::cout << "Usage: ./gpu_my_sol <kmrange1> <file1in> <file1out> ..." << std::endl;
        } else if ((argc - 1) % 3 != 0) {
                std::cout << "Usage: ./gpu_my_sol <kmrange1> <file1in> <file1out> ,,," << std::endl;
        }

        for(int argcID = 1; argcID < argc; argcID += 3) {
                std::vector<float> longit(0);
                std::vector<float> lat(0);
                std::vector<unsigned int> pop(0);
                unsigned int *no_lines_file = 0;
                float *km_range = 0;

                float *longits = 0;
                float *lats = 0;
                unsigned int *pops = 0;
                unsigned int *results = 0;

                cudaMallocManaged(&no_lines_file, sizeof(unsigned int));
                cudaMallocManaged(&km_range, sizeof(float));

                *km_range = atof(argv[argcID]);
                *no_lines_file = fill_data(longit, lat, pop, argv[argcID + 1]);

                cudaMallocManaged(&longits, *no_lines_file * sizeof(float));              
                cudaMallocManaged(&lats, *no_lines_file * sizeof(float));
                cudaMallocManaged(&pops, *no_lines_file * sizeof(unsigned int));
                cudaMallocManaged(&results, *no_lines_file * sizeof(unsigned int));
        
                // reading done    
                cudaMemcpy(longits, longit.data(), *no_lines_file * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(lats, lat.data(), *no_lines_file * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(pops, pop.data(), *no_lines_file * sizeof(unsigned int), cudaMemcpyHostToDevice);
                cudaMemcpy(results, pop.data(), *no_lines_file * sizeof(unsigned int), cudaMemcpyHostToDevice);

                unsigned int block_no = *no_lines_file / BLOCK_SIZE;
                if (*no_lines_file % BLOCK_SIZE) {
                        block_no++;
                }


                master<<<block_no, BLOCK_SIZE>>>(results, longits, lats, pops, *no_lines_file, *km_range);
                if (cudaSuccess != cudaGetLastError()) {
                        printf("pisici\n");
                        return 1;
                }

                // wait for parent to complete
                if (cudaSuccess != cudaDeviceSynchronize()) {
                        printf("caini\n");
                        return 2;
                }
                
                writeResults(argv[argcID + 2], results, *no_lines_file);
        
                cudaFree(no_lines_file);
                cudaFree(km_range);
                cudaFree(longits);
                cudaFree(lats);
                cudaFree(pops);
                cudaFree(results);
        }  
}