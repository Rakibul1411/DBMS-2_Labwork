def find_redo_undo_transactionsID(log_entries):
  
    active_transactions = set()
    committed_transactions = set()
    redo_transactions = []
    undo_transactions = set()
    
    for entry in log_entries:
      
        if "<START" in entry:
          
            transactions_ID = entry.split()[1][:-1]
            active_transactions.add(transactions_ID)

        elif "<COMMIT" in entry:
          
            transactions_ID = entry.split()[1][:-1]

            # if check_checkPoint:
            committed_transactions.add(transactions_ID)
            redo_transactions.append(transactions_ID)
                # commit_check = False

            # if commit_check:
            active_transactions.discard(transactions_ID) 
                # commit_check = True 

        # elif "<CKPT" in entry:
        #     check_checkPoint = True

        elif "<END" in entry:
            break

    # Calculate the undo transactions
    undo_transactions = active_transactions - committed_transactions

    return redo_transactions, undo_transactions

def main():
  
    with open('log.txt', 'r') as file:
        log_entries = file.read().strip().splitlines()  # convert into list

    redo_transactionsID, undo_transactionsID = find_redo_undo_transactionsID(log_entries)

    
    print("Transactions to Redo:", redo_transactionsID)
    print("Transactions to Undo:", undo_transactionsID)

if __name__ == "__main__":
    main()
