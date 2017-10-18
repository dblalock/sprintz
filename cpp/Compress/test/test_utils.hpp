//
//  test_utils.hpp
//  Compress
//
//  Created by DB on 9/21/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

#include "eigen/Eigen"

// #include <stdio.h>

// #include "catch.hpp"

// #include "array_utils.hpp"
// #include "sprintz.h"
// #include "bitpack.h"
// #include "timing_utils.hpp"
// #include "test_utils.hpp"

// #include "debug_utils.hpp" // TODO rm


template<class T> using Vec = Eigen::Array<T, Eigen::Dynamic, 1>;
using Vec_i8 = Vec<int8_t>;
using Vec_u8 = Vec<uint8_t>;
using Vec_i16 = Vec<int16_t>;
using Vec_u16 = Vec<uint16_t>;


template<class T>
void _set_random_bits(T* dest, size_t size, int max_val) {
    T val = static_cast<T>(max_val);
    for (int i = 0; i < size; i += 8) {
        int highest_idx = (i / 8) % 8;
        for (int j = i; j < i + 8; j++) {
            if (j == highest_idx || val == 0) {
                dest[j] = val;
            } else {
                if (val > 0) {
                    dest[j] = rand() % val;
                } else {
                    dest[j] = -(rand() % abs(val));
                }
            }
        }
    }
}

static inline size_t get_filesize(const char* filename) {
    struct stat st;
    if (stat(filename, &st) == -1) {
        perror("Error getting the file size");
        exit(EXIT_FAILURE);
    }
    return st.st_size;
}

static inline std::unique_ptr<uint8_t[]> read_file(std::string path,
    int64_t& nbytes)
{
    int64_t file_size = get_filesize(path.c_str());
    int64_t size = file_size < nbytes || nbytes < 1 ? file_size : nbytes;
    nbytes = size; // nbytes always contains true size
//    auto ret = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
    auto memblock = new uint8_t[size];

    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (f.is_open()) {
        // f.seekg(0, std::ios::beg);
//        f.read(ret.get(), size);
        f.read(reinterpret_cast<char*>(memblock), size);
        f.close();
//        return ret;
        return std::unique_ptr<uint8_t[]>((uint8_t*)memblock);
        // delete[] memblock;
    }
    std::cout << "Unable to open file: '" << path << "'" << std::endl;
    exit(EXIT_FAILURE);
    return std::unique_ptr<uint8_t[]>(nullptr);
}

enum class DatasetName {
    MSRC,
    PAMAP,
    UCI_GAS,
    RAND_1M_0_63,
    CHLORINE
};

typedef struct {
    std::unique_ptr<uint8_t[]> ptr;
    int64_t size_bytes;
    DatasetName name;

    int64_t size() const { return size_bytes; }
    uint8_t* data() const { return ptr.get(); }

} Dataset;

#define DATA_DIR "/Users/davis/Desktop/datasets/compress/rowmajor/uint8/"
#define SYNTH_DATA_DIR "/Users/davis/codez/lzbench/synthetic/"

static inline Dataset read_dataset(DatasetName name,
    int64_t nbytes=-1)
{
    Dataset d;
    d.name = name;
    if (name == DatasetName::MSRC) {
        d.ptr = read_file(DATA_DIR "pamap/pamap.dat", nbytes);
    } else if (name == DatasetName::PAMAP) {
        d.ptr = read_file(DATA_DIR "msrc/msrc.dat", nbytes);
    } else if (name == DatasetName::UCI_GAS) {
        d.ptr = read_file(DATA_DIR "uci_gas/uci_gas.dat", nbytes);
    } else if (name == DatasetName::RAND_1M_0_63) {
        d.ptr = read_file(SYNTH_DATA_DIR "1M_randint_0_63.dat", nbytes);
    } else if (name == DatasetName::CHLORINE) {
        d.ptr = read_file(DATA_DIR "ucr/ChlorineConcentration.dat", nbytes);
    }
    d.size_bytes = nbytes; // written to by read_file
    return d;

    // return read_file("ERROR: invalid dataset! (This can't happen)", nbytes);
}

#endif // TEST_UTILS_HPP
