# Author Muhammad Tehaam
""" Use this code to generate the congestion mapping, first use the experimental routenet 
              to generate predictions and then use this code to map predictions to the specified links! """
import csv
def read_csv_to_dict(file):
    data_dict = {}
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) == 2:
                key, value = row
                if key in data_dict:
                    if not isinstance(data_dict[key], list):
                        data_dict[key] = [data_dict[key]]
                    data_dict[key].append(value)
                else: data_dict[key] = value
    return data_dict
my_dict = dict()
temp = list()
def match_and_sequence(dict1, dict2):
    for path,links in zip(dict1.keys(),dict1.values()):
      link =[int(elem) for elem in links.split(',')]
      for c_links,cong in zip(dict2.keys(),dict2.values()):
        c_links = int(c_links)
        cong = float(cong)
        for L in link:
          if L== c_links:
            temp.append(cong)
      my_dict[path] = sum(temp)/len(temp) if temp else None
      temp.clear()
    return my_dict
file1 = 'links_info.csv'
file2 = 'congestion_predictions.csv'
dict1 = read_csv_to_dict(file1)
dict2 = read_csv_to_dict(file2)

result_sequence = match_and_sequence(dict1, dict2)
print("Result Sequence:")
print(len(result_sequence))


import pandas as pd

def save_dict_to_csv_pandas(data_dict, output_file):
    df = pd.DataFrame(list(data_dict.items()), columns=['Path', 'Cumulative Congestion'])
    df.to_csv(output_file, index=False)

# Example usage

output_csv_file = 'congestion_Model_Results.csv'

save_dict_to_csv_pandas(result_sequence, output_csv_file)
print(f'Data saved to {output_csv_file} successfully.')









########### Uncomment to make appropriate link level information for your data #####################
# import csv
# def duplicate_entries(input_csv, output_csv, iterations):
#     # Read entries from the input CSV file
#     with open(input_csv, 'r') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         header = next(csv_reader)  # Skip the header row
#         entries = [row for row in csv_reader]

#     # Duplicate all entries and append the copies below the original entries
#     duplicated_entries = entries * (iterations + 1)

#     # Write the duplicated entries to the output CSV file
#     with open(output_csv, 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(header)  # Write the header row

#         for duplicated_entry in duplicated_entries:
#             csv_writer.writerow(duplicated_entry)

# # Example usage
# input_csv_file = 'links_info.csv'
# output_csv_file = 'links_info_updated.csv'
# iterations = 200  # Adjust the number of iterations as needed

# duplicate_entries(input_csv_file, output_csv_file, iterations)
# print(f'Data duplicated and written to {output_csv_file} successfully.')
