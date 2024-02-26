def calculate_common_part(input_string, compare_string):
    # Find the common part in both strings
    common_part = ''
    for i in range(min(len(input_string), len(compare_string))):
        if input_string[i] == compare_string[i]:
            common_part += input_string[i]
        else:
            break
    return common_part

def process_string(input_string, common_part):
    # Remove the common part and all occurrences of "TRACK_END"
    #processed_string = input_string.replace(common_part, '').replace('TRACK_END', '')
    processed_string = input_string.replace(common_part, '')
    # Add PIECE_START and TRACK_END if not present
    #if not processed_string.startswith("PIECE_START"):
        #processed_string = "PIECE_START TIME_SIGNATURE=4_4 GENRE=OTHER TRACK_START INST=0 DENSITY=1 " + processed_string
   # if not processed_string.endswith("TRACK_END"):
        #processed_string += " TRACK_END"

    return processed_string

# Example usage
input_string = input("your_input_string_here")
compare_string = input("your_compare_string_here")

common_part = calculate_common_part(input_string, compare_string)
processed_string = process_string(input_string, common_part)
#print(common_part)
print("Processed String:")
print(processed_string)
