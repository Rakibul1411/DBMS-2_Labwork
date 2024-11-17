#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

void writeVectorToFile(const vector<int>& vec, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        for (int value : vec) {
            file << value << "\n";
        }
        file.close();
    } else {
        cerr << "Failed to open file " << filename << "\n";
    }
}

int main() {
    
    int max_cores = omp_get_num_procs();
    cout << "Number of available cores: " << max_cores << endl;
    
    int use_cores = max_cores / 2;
    cout << "Number of uses cores: " << use_cores << endl;
    
    omp_set_num_threads(use_cores);

    
    const int dataSize = 100;
    vector<int> column1(dataSize);
    vector<int> column2(dataSize);
    vector<int> parallel_result;

    
    // #pragma omp parallel for schedule(static, chunk_size)
    for (int i = 0; i < dataSize; ++i) {
        column1[i] = i + 1;
    }

    
    for (int i = 0; i < dataSize; ++i) {
        column2[i] = 2 * i + 1;
    }

    
    unordered_set<int> column2_set(column2.begin(), column2.end());

    vector<vector<int> > core_results(use_cores);

    // Measure time for parallel merge
    auto parallel_start = chrono::high_resolution_clock::now();

    // Parallel matching using OpenMP
    #pragma omp parallel
    {
        int core_id = omp_get_thread_num();
        int num_cores = omp_get_num_threads();
        vector<int>& local_result = core_results[core_id];

        // Calculate the range for this core
        int chunk_size = dataSize / num_cores;
        int start = core_id * chunk_size;
        int end = (core_id == num_cores - 1) ? dataSize : start + chunk_size;

        // Process this core's chunk
        for (int i = start; i < end; ++i) {
            if (column2_set.find(column1[i]) != column2_set.end()) {
                local_result.push_back(column1[i]);
                #pragma omp critical
                {
                    cout << "Match Value: " << column1[i] << " ----> Core used: " << core_id + 1 << endl;
                }
            }
        }
    }

    // Merge results from all cores
    for (const auto& core_result : core_results) {
        parallel_result.insert(parallel_result.end(),
                             core_result.begin(),
                             core_result.end());
    }

    auto parallel_end = chrono::high_resolution_clock::now();
    
    
    sort(parallel_result.begin(), parallel_result.end());

    // Measure time for sequential merge
    vector<int> sequential_result;
    auto sequential_start = chrono::high_resolution_clock::now();

    for (int value : column1) {
        if (column2_set.find(value) != column2_set.end()) {
            sequential_result.push_back(value);
        }
    }

    auto sequential_end = chrono::high_resolution_clock::now();

    
    chrono::duration<double> parallel_duration = parallel_end - parallel_start;
    
    chrono::duration<double> sequential_duration = sequential_end - sequential_start;

    // Print performance metrics
    cout << "\nPerformance Metrics:" << endl;
    cout << "Parallel execution time (" << use_cores << " cores): "
         << parallel_duration.count() << " seconds" << endl;
    
    cout << "Sequential execution time: "
         << sequential_duration.count() << " seconds" << endl;
    cout << "Speedup: "
         << sequential_duration.count() / parallel_duration.count() << "x" << endl;

    // Print core utilization
    cout << "\nCore Utilization:" << endl;
    for (int i = 0; i < use_cores; ++i) {
        cout << "Core " << i+1 << " processed "
             << core_results[i].size() << " matches" << endl;
    }

    
    writeVectorToFile(column1, "column1.txt");
    writeVectorToFile(column2, "column2.txt");
    writeVectorToFile(parallel_result, "matching_values.txt");

    cout << "\nResults:" << endl;
    cout << "Total matches found: " << parallel_result.size() << endl;
    
    cout << "First 10 matches: ";
    for (int i = 0; i < min(10, static_cast<int>(parallel_result.size())); ++i) {
        cout << parallel_result[i] << " ";
    }
    
    cout << "\n\nData has been written to files" << endl;

    return 0;
}
