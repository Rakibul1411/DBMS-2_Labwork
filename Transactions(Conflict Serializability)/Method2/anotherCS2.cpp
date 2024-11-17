#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

struct Operation {
    int transactionId;
    char operationType;
    char dataItem;

    Operation(int id, char type, char item) 
        : transactionId(id), operationType(type), dataItem(item) {}
};


bool detectCycle(int node, unordered_map<int, vector<int> >& graph, unordered_set<int>& visited, unordered_set<int>& recursionStack) {

    if (recursionStack.count(node)) { 
      return true;
    } // Cycle found

    if (visited.count(node)) {
      return false;
    }

    visited.insert(node);
    recursionStack.insert(node);

    for (int neighbor : graph[node]) {
        if (detectCycle(neighbor, graph, visited, recursionStack)) 
            return true;
    }

    recursionStack.erase(node);
    return false;

}


bool isConflictSerializable(const vector<vector<string> >& schedule) {
    vector<Operation> operations;
    int numTransactions = schedule.size();

    
    for (size_t step = 1; step < schedule[0].size(); ++step) {
        for (int transactionIdx = 0; transactionIdx < numTransactions; ++transactionIdx) {

            const string& action = schedule[transactionIdx][step];
            
            if (action != "-") {
                operations.emplace_back(
                    transactionIdx + 1, 
                    action[0],         
                    action[2]          
                );
            }
        }
        
    }


    unordered_map<int, vector<int> > precedenceGraph;

    for (size_t i = 0; i < operations.size(); ++i) {
        for (size_t j = i + 1; j < operations.size(); ++j) {
            Operation& op1 = operations[i];
            Operation& op2 = operations[j];

            if (op1.transactionId != op2.transactionId && op1.dataItem == op2.dataItem) {
                if ((op1.operationType == 'W' && op2.operationType == 'R') || 
                    (op1.operationType == 'R' && op2.operationType == 'W') || 
                    (op1.operationType == 'W' && op2.operationType == 'W')) 
                {
                    precedenceGraph[op1.transactionId].push_back(op2.transactionId);
                }
            }
        }
    }

    
    unordered_set<int> visited, recursionStack;

    for (int t = 1; t <= numTransactions; ++t) {
        if (detectCycle(t, precedenceGraph, visited, recursionStack)) 
            return false; // Cycle detected
    }

    return true; // No cycles found, so it's conflict-serializable
}


vector<vector<string> > parseScheduleFile(const string& filename) {
    ifstream inputFile(filename);
    vector<vector<string> > schedule;
    string line;

    if (!inputFile) {
        cerr << "Error opening file: " << filename << endl;
        return schedule;
    }

    while (getline(inputFile, line)) {
        stringstream ss(line);
        vector<string> transaction;
        string operation;

        while (getline(ss, operation, ',')) {
            operation.erase(0, operation.find_first_not_of(" "));
            operation.erase(operation.find_last_not_of(" ") + 1);
            transaction.push_back(operation);
        }
        schedule.push_back(transaction);
    }
    inputFile.close();
    return schedule;
}

int main() {
    string filename = "schedule2.txt";
    vector<vector<string> > schedule = parseScheduleFile(filename);

    if (schedule.empty()) {
        cerr << "No schedule data found in file." << endl;
        return 1;
    }

    if (isConflictSerializable(schedule)) {
      cout << endl << endl;
      cout << "The schedule is conflict serializable." << endl << endl;
    } 
    else {
      cout << endl << endl;
      cout << "The schedule is not conflict serializable." << endl << endl;
    }

    return 0;
}
