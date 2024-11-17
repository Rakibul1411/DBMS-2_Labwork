#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono> // For measuring time

using namespace std;

mutex result_mutex;

// Function to find matches in a single partition
void findMatchesInPartition(const vector<int>& partition, const unordered_set<int>& column2_set, vector<int>& result) {
    vector<int> local_result;
    for (int value : partition) {
        if (column2_set.find(value) != column2_set.end()) {
            local_result.push_back(value);
        }
    }

    lock_guard<mutex> lock(result_mutex);
    result.insert(result.end(), local_result.begin(), local_result.end());
}

// Function to write a vector to a file
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
    srand(std::time(0));

    int dataSize = 10000;
    vector<int> column1(dataSize);
    for (int i = 0; i < dataSize; ++i) {
        column1[i] = i + 1;
    }

    vector<int> column2(dataSize);
    int indx = 0;
    for (int i = 1; i < dataSize * 2; i = i + 2) {
        column2[indx] = i;
        ++indx;
    }

    unordered_set<int> column2_set(column2.begin(), column2.end());

    int partitionSize = dataSize / 4;
    vector<vector<int> > partitions;
    for (int i = 0; i < 4; ++i) {
        partitions.push_back(vector<int>(column1.begin() + i * partitionSize, column1.begin() + (i + 1) * partitionSize));
    }

    vector<int> parallel_result;

    // Measure time for parallel merge
    auto parallel_start = chrono::high_resolution_clock::now();

    vector<thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.push_back(thread(findMatchesInPartition, ref(partitions[i]), ref(column2_set), ref(parallel_result)));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto parallel_end = chrono::high_resolution_clock::now();
    chrono::duration<double> parallel_duration = parallel_end - parallel_start;
    cout << "Parallel merge time (4 threads): " << parallel_duration.count() << " seconds" << endl;

    sort(parallel_result.begin(), parallel_result.end());

    // Measure time for sequential merge
    vector<int> sequential_result;
    auto sequential_start = chrono::high_resolution_clock::now();

    for (int i = 0; i < column1.size(); i++) {
        for (int j = 0; j < column2.size(); j++) {
            if (column1[i] == column2[j])
                sequential_result.push_back(column1[i]);
        }
    }

    auto sequential_end = chrono::high_resolution_clock::now();
    chrono::duration<double> sequential_duration = sequential_end - sequential_start;
    cout << "Sequential merge time (1 thread): " << sequential_duration.count() << " seconds" << endl;

    // Write column1, column2, and results to files
    writeVectorToFile(column1, "column1.txt");
    writeVectorToFile(column2, "column2.txt");
    writeVectorToFile(parallel_result, "matching_values.txt");

    cout << "Data has been written to column1.txt, column2.txt, and matching_values.txt" << endl;

    return 0;
}
