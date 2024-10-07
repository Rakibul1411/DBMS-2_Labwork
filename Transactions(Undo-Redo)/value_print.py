def find_redo_undo_transactionsID(log_entries):
    active_transactions = set()
    committed_transactions = set()
    redo_transactions = []
    undo_transactions = set()
    check_checkPoint = False
    commit_check = True
    variable_values_old = {}
    variable_values_new = {}

    for entry in log_entries:
        if "<START" in entry:
            transactions_ID = entry.split()[1][:-1]
            active_transactions.add(transactions_ID)

        elif "<COMMIT" in entry:
            transactions_ID = entry.split()[1][:-1]

            if check_checkPoint:
                committed_transactions.add(transactions_ID)
                redo_transactions.append(transactions_ID)
                commit_check = False

            if commit_check:
                print("Ignore: ", transactions_ID)
                active_transactions.discard(transactions_ID)
                commit_check = True

        elif "<CKPT" in entry:
            check_checkPoint = True

        elif "<END" in entry:
            break

        else:
            parts = entry.split()
            if len(parts) == 4:
                transactions_ID = parts[0][1:]
                variable = parts[1]
                value1 = parts[2]
                value2 = parts[3][:-1]

                # Ensure the transactions_ID exists in the dictionary
                if transactions_ID not in variable_values_old:
                    variable_values_old[transactions_ID] = {}
                if transactions_ID not in variable_values_new:
                    variable_values_new[transactions_ID] = {}

                # Now safely set the variable values
                variable_values_old[transactions_ID][variable] = value1
                variable_values_new[transactions_ID][variable] = value2

    # Calculate the undo transactions
    undo_transactions = active_transactions - committed_transactions

    return redo_transactions, undo_transactions, variable_values_old, variable_values_new


def main():
    with open('input.txt', 'r') as file:
        log_entries = file.read().strip().splitlines()  # convert into list

    redo_transactionsID, undo_transactionsID, variable_values_old, variable_values_new = find_redo_undo_transactionsID(log_entries)

    print("Transactions to Redo:", redo_transactionsID)
    print("Transactions to Undo:", undo_transactionsID)

    final_values = {}

    # Print values of A, B, C, D
    for variable in ['A', 'B', 'C', 'D']:
        for trans_id in redo_transactionsID:
            if variable in variable_values_new.get(trans_id, {}):
                final_values[variable] = variable_values_new[trans_id][variable]

        # if variable not in final_values:
        for trans_id in undo_transactionsID:
            if variable in variable_values_old.get(trans_id, {}):
                final_values[variable] = variable_values_old[trans_id][variable]

    for variable in ['A', 'B', 'C', 'D']:
        print(f"Final Value of {variable}:", final_values.get(variable, "Not updated"))


if __name__ == "__main__":
    main()
