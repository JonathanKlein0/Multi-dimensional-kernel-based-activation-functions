import ast
import numpy as np

def combine_array(arr1, arr2):
    new_arr = [arr1[0], arr1[1] + arr2[1], arr1[2] + arr2[2], 0, 0, 0, 0]
    new_arr[3] = float(np.mean(new_arr[1]))
    new_arr[4] = float(np.std(new_arr[1]))
    new_arr[5] = float(np.mean(new_arr[2]))
    new_arr[6] = float(np.std(new_arr[2]))
    return new_arr

def process_file(file_name):
    with open(file_name, "r") as f_in:
        lines = f_in.readlines()

    # first line unchanged
    header = lines[0]
    # drop final empty line
    data_lines = [l for l in lines[1:] if l.strip()]

    n = len(data_lines)
    assert n % 2 == 0, "Number of data lines must be even"

    half = n // 2
    new_lines = []

    for k in range(half):
        # split "word: array"
        word1, arr1_str = data_lines[k].split(": ", 1)
        word2, arr2_str = data_lines[k + half].split(": ", 1)

        assert word1 == word2, "Mismatched words"

        arr1 = ast.literal_eval(arr1_str)
        arr2 = ast.literal_eval(arr2_str)

        result = combine_array(arr1, arr2)

        new_lines.append(f"{word1}: {result}\n")

    with open(file_name, "w") as f_out:
        f_out.write(header)
        f_out.writelines(new_lines)
