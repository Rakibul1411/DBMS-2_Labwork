#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
using namespace std;

struct Operation {
    int transaction;
    char type;
    char dataItem;

    Operation(int t, char opType, char item) : transaction(t), type(opType), dataItem(item) {}
};

bool hasCycle(int node, unordered_map<int, vector<int> >& graph, unordered_set<int>& visited, unordered_set<int>& stack) {
    if (stack.find(node) != stack.end()) {
        return true; // Cycle detected
    }

    if (visited.find(node) != visited.end()) {
        return false;
    }

    visited.insert(node);
    stack.insert(node);

    for (int neighbor : graph[node]) {
        if (hasCycle(neighbor, graph, visited, stack)) {
            return true;
        }
    }

    stack.erase(node);
    return false;
}

bool isConflictSerializable(const vector<vector<string> >& schedule) {
    vector<Operation> operations;
    int numTransactions = schedule.size();

    for (int i = 1; i < schedule[0].size(); i++) {
        for (int t = 0; t < numTransactions; t++) {
            if (schedule[t][i] != "-") {
                char type = schedule[t][i][0];
                char dataItem = schedule[t][i][2];
                // Use emplace_back for in-place construction
                operations.emplace_back(t + 1, type, dataItem);
            }
        }
    }

    unordered_map<int, vector<int> > precedenceGraph;
    for (size_t i = 0; i < operations.size(); i++) {
        for (size_t j = i + 1; j < operations.size(); j++) {
            if (operations[i].transaction != operations[j].transaction && operations[i].dataItem == operations[j].dataItem) {
                if ((operations[i].type == 'W' && operations[j].type == 'R') ||
                    (operations[i].type == 'R' && operations[j].type == 'W') ||
                    (operations[i].type == 'W' && operations[j].type == 'W')) {
                    precedenceGraph[operations[i].transaction].push_back(operations[j].transaction);
                }
            }
        }
    }

    unordered_set<int> visited, stack;
    for (int t = 1; t <= numTransactions; t++) {
        if (hasCycle(t, precedenceGraph, visited, stack)) {
            return false; // Cycle detected
        }
    }

    return true; // No cycles, hence conflict serializable
}

int main() {
    vector<vector<string> > schedule;
    string line;

    ifstream inputFile("schedule3.txt");
    if (!inputFile) {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    while (getline(inputFile, line)) {
    stringstream ss(line);
    vector<string> transaction;
    string operation;

    while (getline(ss, operation, ',')) {
        operation.erase(0, operation.find_first_not_of(" \t"));
        operation.erase(operation.find_last_not_of(" \t") + 1);

        transaction.push_back(operation);
    }

        schedule.push_back(transaction);
    }


    inputFile.close();

    if (isConflictSerializable(schedule)) {
        cout << "The schedule is conflict serializable." << endl;
    } else {
        cout << "The schedule is not conflict serializable." << endl;
    }

    return 0;
}
