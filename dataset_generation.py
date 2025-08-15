

import csv

def calculate_times(row):
    arrival_time = float(row['Arrival Time'])
    deadline = float(row['Deadline'])
    duration = float(row['Duration'])
    E2E_delay = float(row['Actual Delay'])
  
    print(arrival_time,deadline,duration,E2E_delay)
    completion_time = arrival_time + duration + E2E_delay
    if completion_time<=deadline:
      
      return {'Completion Time': completion_time, 'Deadline Will be Met': 1}

    else:
      return {'Completion Time': completion_time, 'Deadline Will be Met': 0}
rows = []
total_flows_accepted= 0
with open('ground_truth_dataset.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        times = calculate_times(row)
        if times['Deadline Will be Met']==1: total_flows_accepted = total_flows_accepted+1
        row.update(times)
        rows.append(row)
fieldnames = list(rows[0].keys())
with open('dataset.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    print("DataSet Generated!")
print("Total Flows that can be accepted:",total_flows_accepted,"out of: ",len(rows))














########################## Prototype 2 ######################################################

# import pandas as pd

# # Read the CSV file
# df = pd.read_csv('ground_truth_dataset.csv')
# #df['Deadline Will be Met']
# # Iterate through each row
# def find_deadline(row):
#   for index, row in df.iterrows():
#       arrival_time = row['Arrival Time']
#       duration = row['Duration']
#       end_to_end_delay = row['Actual Delay']
#     #remaining_time = 30
#     # Add more variables for additional columns as needed
#       if end_to_end_delay is not None:
#              completion_time = arrival_time + end_to_end_delay + duration
#              deadline = row['Deadline']
#              print("processed!")
#              if completion_time <= deadline:
#                  #row['Deadline Will be Met'] = 1
#                  #row['Completion Time'] = completion_time
#                  return pd.Series([completion_time, 1], index=['Completion Time', 'Deadline Will be Met'])
#              else:
#               return pd.Series([completion_time, 0], index=['Completion Time', 'Deadline Will be Met'])
#               #return completion_time
#                 # row['Deadline Will be Met'] = 0 # False
#                 # row['Completion Time']= completion_time
#     # Apply your logic to the variables here
#     # For example, you can perform calculations, comparisons, etc.

# df[['Completion Time', 'Deadline Will be Met']] = df.apply(find_deadline, axis=1)

# df.to_csv('ground_truth_dataset.csv', index=False)
#     # Print or do something with the variables
#     #print(f'Row {index+1}: Column1 data = {column1_data}, Column2 data = {column2_data}')



########################## Prototype 1 ######################################################



# import pandas as pd

# def predict_deadline_meeting_with_arrival_from_csv(file_path, tFA):
#     df = pd.read_csv(file_path)
#     print(len(df))
#     end_to_end_delays = dict(zip(zip(df['Source'], df['Destination']), df['Actual Delay']))
    
#     deadline_met_dict = {}
    
#     for _, row in df.iterrows():
#         arrival_time, duration, volume, source_node, destination_node,remaining_time = row['Arrival Time'], row['Duration'], row['Volume'], row['Source'], row['Destination'],row['Deadline']
#         end_to_end_delay = end_to_end_delays.get((source_node, destination_node), None)
#         print(end_to_end_delay)
#         if end_to_end_delay is not None:
#             completion_time = arrival_time + end_to_end_delay + duration
           
#             if completion_time <= remaining_time:
#                 deadline_met_dict[(source_node, destination_node)] = 1 # True
#                 tFA+=1
#             else:
#                 deadline_met_dict[(source_node, destination_node)] = 0 # False

#     return deadline_met_dict,tFA

# file_path = 'ground_truth_dataset.csv'  
# # remaining_time = 30 
# total_flow_acceptance = 0

# i=0

# import csv 
# existing_data=[]
# with open('ground_truth_dataset.csv',mode='r') as file:
#   reader = csv.DictReader(file)
#   existing_data = [row for row in reader]
# new_columns = ["Deadline Will be Met"]

# deadline_met_dict = predict_deadline_meeting_with_arrival_from_csv(file_path,total_flow_acceptance)


# print(len(existing_data),len(deadline_met_dict[0]))

# for (source_dest,deadline_met),row in zip((deadline_met_dict[0].items()),existing_data):
#     row["Deadline Will be Met"]= deadline_met
#     print(f"For Source-Destination Pair {source_dest}: Deadline Will be Met: {deadline_met}")
    
#     i+=1


# with open('dataset.csv',mode='w',newline='') as file:
#   fieldnames = list(existing_data[0].keys())+list(new_columns)
#   writer = csv.DictWriter(file,fieldnames=fieldnames)
#   writer.writeheader()
#   for row in existing_data:
#     writer.writerow(row)

# print("Total Flow Accepted: ",deadline_met_dict[1],"Out of: ",i)
# print("Flow Acceptance Ratio: ",(deadline_met_dict[1]/i)*100,"%")


