#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <iomanip>
#include <iterator>

using namespace std;

vector<set<string>> generateCombinations(const set<string>& items) {
    vector<set<string>> allCombinations;
    int n = items.size();
    vector<string> itemVec(items.begin(), items.end());

    for (int i = 1; i < (1 << n) - 1; ++i) {
        set<string> subset;
        for (int j = 0; j < n; ++j) {
            if (i & (1 << j)) {
                subset.insert(itemVec[j]);
            }
        }
        allCombinations.push_back(subset);
    }
    return allCombinations;
}


void generateAssociationRules(const map<set<string>, int>& frequentItemsets, const map<set<string>, int>& track, double minConfidence, int noTrans) {
    cout << "\nAssociation Rules\n";
    cout << "-----------------------------------------------\n";
    cout << "Rule                   | Confidence(%) | Lift\n";
    cout << "-----------------------------------------------\n";

    for (const auto& itemset : track) {
        int itemsetSupport = itemset.second;

        vector<set<string>> combinations = generateCombinations(itemset.first);

        for (const auto& subset : combinations) {
            set<string> difference;
            set_difference(itemset.first.begin(), itemset.first.end(),
                           subset.begin(), subset.end(),
                           inserter(difference, difference.begin()));

            if (frequentItemsets.find(subset) != frequentItemsets.end()) {
                int subsetSupport = frequentItemsets.at(subset);
                double confidence = static_cast<double>(itemsetSupport) / subsetSupport;

                if (frequentItemsets.find(difference) != frequentItemsets.end()) {
                    int differenceSupport = frequentItemsets.at(difference);
                    double lift = (confidence * noTrans) / (static_cast<double>(differenceSupport));

                    if (confidence >= minConfidence) {
                        for (const auto& item : subset) cout << item << " ";
                        cout << " -> ";
                        for (const auto& item : difference) cout << item << " ";
                        cout << "   " << setw(15) << fixed << setprecision(2) << (confidence * 100)
                             << setw(15) << fixed << setprecision(2) << lift << endl;
                    }
                }
            }
        }
    }
}


void countSupport(map<set<string>, int>& candidates, const vector<set<string>>& transactions) {
    for (auto& candidate : candidates) {
        candidate.second = 0;
        for (const auto& transaction : transactions) {
            if (includes(transaction.begin(), transaction.end(), candidate.first.begin(), candidate.first.end())) {
                candidate.second++;
            }
        }
    }
}


map<set<string>, int> generateCandidates(const set<set<string>>& frequentItemsets, int size) {
    map<set<string>, int> candidates;
    for (auto it1 = frequentItemsets.begin(); it1 != frequentItemsets.end(); ++it1) {
        for (auto it2 = next(it1); it2 != frequentItemsets.end(); ++it2) {
            set<string> candidate(it1->begin(), it1->end());
            candidate.insert(it2->begin(), it2->end());
            if (candidate.size() == size) {
                candidates[candidate] = 0;
            }
        }
    }
    return candidates;
}


map<set<string>, int> filterCandidates(const map<set<string>, int>& candidates, int minSupport) {
    map<set<string>, int> frequentItemsetsWithSupport;
    for (const auto& candidate : candidates) {
        if (candidate.second >= minSupport) {
            frequentItemsetsWithSupport[candidate.first] = candidate.second;
        }
    }
    return frequentItemsetsWithSupport;
}


void apriori(const vector<set<string>>& transactions, int minSupport, int noTrans) {
    map<set<string>, int> allFrequentItemsets;
    set<set<string>> frequentItemsets;
    map<set<string>, int> track;

    map<string, int> itemCount;
    for (const auto& transaction : transactions) {
        for (const auto& item : transaction) {
            itemCount[item]++;
        }
    }

    cout << "Number of transactions: " << transactions.size() << endl;
    cout << "Transactions   |   ItemSets" << endl;
    cout << "-----------------------------" << endl;
    for (size_t i = 0; i < transactions.size(); ++i) {
        cout << "T" << i + 1 << ":              [ ";
        for (const auto& item : transactions[i]) {
            cout << item << " ";
        }
        cout << "]" << endl;
    }

    cout << "\nStep-1\nC-1\nItemSets   |   Support_Count" << endl;
    cout << "----------------------------" << endl;
    for(const auto& item : itemCount){
        cout << "[" << item.first << "]                  " << item.second << endl;
    }

    cout << endl;


    cout << "\nStep-2\nL-1\nItemSets   |   Support_Count" << endl;
    cout << "----------------------------" << endl;
    for (const auto& item : itemCount) {
        if (item.second >= minSupport) {
            set<string> singleItemSet;
            singleItemSet.insert(item.first);
            frequentItemsets.insert(singleItemSet);
            allFrequentItemsets[singleItemSet] = item.second;
            cout << "[" << item.first << "]                  " << item.second << endl;
        }
    }

    int k = 2;
    while (!frequentItemsets.empty() && frequentItemsets.size() > 1) {
        map<set<string>, int> candidates = generateCandidates(frequentItemsets, k);

        countSupport(candidates, transactions);

        cout << "\nC-" << k << "\nItemSets   |   Support_Count" << endl;
        cout << "----------------------------" << endl;
        for (const auto& candidate : candidates) {
            cout << "[ ";
            for (const auto& item : candidate.first) {
                cout << item << " ";
            }
            cout << "]         " << candidate.second << endl;
        }

        map<set<string>, int> frequentItemsetsWithSupport = filterCandidates(candidates, minSupport);

        cout << endl;

        if(!frequentItemsetsWithSupport.empty()) {
            cout << "\nStep-" << k+1 << "\nL-" << k << "\nItemSets   |   Support_Count" << endl;
            cout << "----------------------------" << endl;

            frequentItemsets.clear();
            track.clear();
            for (const auto& itemset : frequentItemsetsWithSupport) {
                track[itemset.first] = itemset.second;

                frequentItemsets.insert(itemset.first);
                allFrequentItemsets[itemset.first] = itemset.second;

                cout << "[ ";
                for (const auto& item : itemset.first) {
                    cout << item << " ";
                }
                cout << "]         " << itemset.second << endl;
            }
        }
        else {
            break;
        }

        k++;
    }

    cout << "\nItemSets    |    Support_Count" << endl;
    cout << "-----------------------------" << endl;

    for (const auto& itemset : track) {
        cout << "[ ";

        for (const auto& item : itemset.first) {
            cout << item << " ";
        }

        cout << "]" << setw(20 - itemset.first.size() * 3) << right << itemset.second << endl;
    }

    cout << endl;

    cout << "\nItemSets    |    Support_Count" << endl;
    cout << "-----------------------------" << endl;

    for (const auto& itemset : allFrequentItemsets) {
        cout << "[ ";

        for (const auto& item : itemset.first) {
            cout << item << " ";
        }

        cout << "]" << setw(20 - itemset.first.size() * 3) << right << itemset.second << endl;
    }

    cout << endl << endl;

    double minConfidence; // New variable for minimum confidence
    cout << "Enter the Minimum Confidence (0.0 - 1.0): ";
    cin >> minConfidence; // Input for minimum confidence

    generateAssociationRules(allFrequentItemsets, track, minConfidence, noTrans);

}


vector<set<string>> parseInputFile(const string& filename) {
    ifstream inputFile(filename);
    vector<set<string>> transactions;
    string line;

    if (!inputFile) {
        cerr << "Error opening file: " << filename << endl;
        return transactions;
    }

    while (getline(inputFile, line)) {
        set<string> transaction;
        size_t start = line.find('{') + 1;
        size_t end = line.find('}');
        string items = line.substr(start, end - start);
        size_t pos = 0;

        while ((pos = items.find(',')) != string::npos) {
            transaction.insert(items.substr(0, pos));
            items.erase(0, pos + 1);
        }
        transaction.insert(items);
        transactions.push_back(transaction);
    }

    inputFile.close();
    return transactions;
}

int main() {
    string filename = "transaction1.txt";
    int minSupport;

    cout << "Enter the Minimum Support Count: ";
    cin >> minSupport;

    vector<set<string>> transactions = parseInputFile(filename);

    apriori(transactions, minSupport, transactions.size());

    return 0;
}
