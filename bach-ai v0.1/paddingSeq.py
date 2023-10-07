# padding the lists to fit the model - Bach AI v0.1

# 07/10/2023 - Jakob De Vreese

def padding_lists(fugue_list, theme_list):
    
    # Give Feedback
    print("")
    print("Padding the lists to fit the model...")
    print("")
    print("Quite hard to explain - lets just say it is necessary for the model to work")
    print("")
    print("")
    
    # Transform data for training
    # pad sequences to max length
    # determine max length of fugue
    max_fugue_len = max(len(fugue) for fugue in fugue_list)

    # Pad sequences to max length by repeating the sequence
    theme_list_padded = [row * (max_fugue_len // len(row)) + row[:max_fugue_len % len(row)] for row in theme_list]
    fugue_list_padded = [row * (max_fugue_len // len(row)) + row[:max_fugue_len % len(row)] for row in fugue_list]

     # Give feedback
    print("")
    print("Padded sequences fugues to max length: "+ str(max_fugue_len))
    print("Padded sequences themes to max length: "+ str(max_fugue_len))
    print("")
    print("numer of rows theme list: " + str(len(theme_list)))
    print("numer of rows fugue list: " + str(len(fugue_list)))
    print("")
    print("number of cells in a row theme list: " + str(len(theme_list_padded[0])))
    print("number of cells in a row fugue list: " + str(len(fugue_list_padded[0])))
    print("")
    print("")
    print("")
    print("Padding done!")
    print("pfew, that was a lot of work!")
    print("")
    print("Step 2 done, we can start the real work now!")
    print("")
    print("")
    print("")

    return fugue_list_padded, theme_list_padded

