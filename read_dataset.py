"""
   Copyright 2020 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import tensorflow as tf
import csv
from datanetAPI import DatanetAPI

mydict=dict()
data_to_return = list()
POLICIES = np.array(['WFQ', 'SP', 'DRR'])
WEIGHTS  = np.array(['90,5,5','80,10,10','75,25,5','70,20,10','65,25,10','60,30,10','50,40,10','33.3,33.3,33.3'])
#when reading the evaluation data-set (i.e., generating the prediction) replace the line above by the line below!
#WEIGHTS  = np.array(['90,5,5','80,10,10','75,25,5','70,20,10','65,25,10','60,30,10','50,40,10','33.3,33.3,33.4'])


def policy(graph, node):
    weight=-8
    pol =   np.where(POLICIES==graph.nodes[node]['schedulingPolicy'])[0][0]
    if (pol!=1):
        weight=np.where(WEIGHTS==graph.nodes[node]['schedulingWeights'])[0][0]+1
    return (pol, weight/8.)   #for 'SP' is always -1, otherwise between ]0.,1.]

################################ Custom Function ############################################
def return_corresponding_data(data_dir, shuffle = False):
  return mydict,data_to_return
########################################################################################################################################



import networkx as nx
import matplotlib.pyplot as plt

def generator(data_dir, shuffle = False,generat_data=False):
    """This function uses the provided API to read the data and returns
       and returns the different selected features.

    Args:
        data_dir (string): Path of the data directory.
        shuffle (string): If true, the data is shuffled before being processed.

    Returns:
        tuple: The first element contains a dictionary with the following keys:
            - bandwith
            - packets
            - link_capacity
            - links
            - paths
            - sequences
            - n_links, n_paths
            The second element contains the source-destination delay
    """
    tool = DatanetAPI(data_dir, [], shuffle)
    ii=0
    item= 0
    it = iter(tool)
    for sample in it:
        ###################
        #  EXTRACT PATHS  #
        ###################
        routing = sample.get_routing_matrix()
        traffic = sample.get_traffic_matrix()

       # print("TRAFFIC:", traffic)
        nodes = len(routing)
     
        # Remove diagonal from matrix
        paths = routing[~np.eye(routing.shape[0], dtype=bool)].reshape(routing.shape[0], -1)
        paths = paths.flatten()
       
      

        ###################
        #  EXTRACT LINKS  #
        ###################
        g = sample.get_topology_object()



                    
        cap_mat         = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)
        tx_queue_mat    = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)
        tx_weight_mat   = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)
        
        for node in range(g.number_of_nodes()):
            for adj in g[node]:
                cap_mat[node, adj] = g[node][adj][0]['bandwidth']*1.E-5  #BEWARE : SCALING  !
                tx_queue_mat[node, adj] = policy(g,node)[0]
                tx_weight_mat[node, adj] = policy(g,node)[1]

        links = np.where(np.ravel(cap_mat) != None)[0].tolist()
        link_capacities   = (np.ravel(cap_mat)[links]).tolist()
        tx_policies       = (np.ravel(tx_queue_mat)[links]).tolist()
        tx_weights         = (np.ravel(tx_weight_mat)[links]).tolist()
        
        ids = list(range(len(links)))
        links_id = dict(zip(links, ids))

        path_ids = []
        weight_ids = []
        pol_ids = []
        for path in paths:
            new_path = []
            for i in range(0, len(path) - 1):
                src = path[i]
                dst = path[i + 1]         
                new_path.append(links_id[src * nodes + dst])

            path_ids.append(new_path)
        ###################
        #   MAKE INDICES  #
        ###################

        link_indices  = []
        path_indices  = []
        sequ_indices  = []
        segment = 0
        for i in range(len(path_ids)):
            p=path_ids[i]
            link_indices += p
            path_indices += len(p) * [segment]
            sequ_indices += list(range(len(p)))
            segment += 1
       # print(len(path_indices))
       # print(len(link_indices))
        # Remove diagonal from matrix
        traffic = traffic[~np.eye(traffic.shape[0], dtype=bool)].reshape(traffic.shape[0], -1)

        result = sample.get_performance_matrix()
        # Remove diagonal from matrix
        result = result[~np.eye(result.shape[0], dtype=bool)].reshape(result.shape[0], -1)

        avg_bw = []
        pkts_gen = []
        delay = []
        tos = []
        AvgPkS = []

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                flow = traffic[i, j]['Flows'][0]
                avg_bw.append(flow['AvgBw']*1.E-3)
                tos.append(float(flow['ToS']))
                AvgPkS.append(flow['SizeDistParams']['AvgPktSize']*1.E-4)
                pkts_gen.append(flow['PktsGen'])
                d = result[i, j]['AggInfo']['AvgDelay']
                delay.append(d)

        n_paths = len(path_ids)
        n_links = max(max(path_ids)) + 1

        # def print_path_links(path_id):
        #   path = paths[path_id]
        #   print(f"Path {path_id + 1}:")
        #   for node in range(len(path) - 1):
        #       src, dst = path[node], path[node + 1]
        #       link_id = links_id[src * nodes + dst]
        #       link_info = f"Link {link_id}: Capacity={link_capacities[link_id]}, Queue Policy={tx_policies[link_id]}, Weight={tx_weights[link_id]}"
        #       print(link_info)

        #   # Print path and links for each path
        # for i in range(len(paths)):
        #       print_path_links(i)
        #       print()  # Add a newline between paths for better readabilit

      #  print(n_paths,"Number of Paths!!!!!!!")
       # print(n_links,"Number of Paths!!!!!!!")
       # print(paths)
        G = nx.DiGraph()

        # Add nodes to the graph
        for node in range(nodes):
            G.add_node(node)

        # Add edges (links) to the graph
        for link in links:
            src, dst = divmod(link, nodes)
        
        # mydict={}
        #print(len(link_capacities))
        #print(len(links))
        for path in paths:
          temp =[]
          for node in range(len(path)-1):
            psrc, pdst = path[node], path[node + 1]
            for link,cap in zip(links,link_capacities):
              src,dst = divmod(link,nodes)
              if psrc==src and pdst==dst:
               # print(path,":",link,":",cap)
                temp.append(link)
                item=item+1
          mydict[tuple(path)]=temp
          #print(mydict)
          # temp.clear()
         # print(item)
        #  print(mydict)
            #G.add_edge(src, dst)
        #print(len(paths))
        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black", font_weight="bold", edge_color="gray", linewidths=0.5, arrowsize=10)
        plt.title("Network Topology")


       # plt.show()

        #print(nodes,"NODES")
        #print(links,"LINKS")



        for link in links:
          data_to_return.append(link)



        lambda_value = 0.1  # Parameter for exponential distribution (controls mean arrival time)
        lower_limit_duration = 5  # Lower limit for duration
        upper_limit_duration = 30  # Upper limit for duration
        lower_limit_volume = 100  # Lower limit for volume
        upper_limit_volume = 1000  # Upper limit for volume
      #  print(link_indices)
       # print(len(link_indices))
        if generat_data==True:
          csv_file = 'ground_truth_dataset.csv'
          keys = ["Arrival Time","Duration","Volume","Source", "Destination", "Path", "Average Packets", "Average Bandwidth","Deadline","Actual Delay"]
          with open(csv_file, 'a', newline='') as file:
        # Create a CSV writer object
            writer = csv.DictWriter(file, fieldnames=keys)
          #  writer.writeheader()
           # writer = csv.DictWriter(file,fieldnames=keys)
            for path,avgBW,pktsGen,dly,ts,avgPKS in zip(paths,avg_bw,pkts_gen,delay,tos,AvgPkS):
              src = path[0]
              dst = path[len(path)-1]
              arrival_time = np.random.exponential(scale=1/lambda_value)
              duration = np.random.uniform(lower_limit_duration, upper_limit_duration)
              volume = np.random.uniform(lower_limit_volume, upper_limit_volume)
              deadline = np.random.uniform(30,40) 
          
              row_dict = {
                "Actual Delay": dly,
                "Source": src,
                "Destination": dst,
                "Path": path,
                "Average Packets": avgPKS,
                "Average Bandwidth": avgBW,
                "Arrival Time": arrival_time,
                "Duration": duration,
                "Volume": volume,
                "Deadline":deadline
                             
              }
              writer.writerow(row_dict)
              print("Ground Truth Updated!")
              

            
        
       # print("Total Paths: ",len(paths))
        #print("Total Links: ",len(pkts_gen))
  ####################  ACTUAL PRINTING AREA   ########################################      
      #  for path,packs,abw in zip(paths,AvgPkS,avg_bw):
       #   src = path[0]
        #  dst = path[len(path)-1]
  
         # print(ii,":","SOURCE:",src,"DESTINATION:",dst,"PATH:",path,"TOPOLOGY:",g,"Average Packets:",packs,"Average Bandwidth:",abw)
          
          #ii+=1
        
 ######################################################   

        # print("Length of avg_bw:",len(avg_bw))
        # print("Length of pkts_gen:",len(pkts_gen))
        # print("Length of tos:",len(tos))
        # print("Length of AvgPkS:",len(AvgPkS))
        # print("Length of link_capacities:",len(link_capacities))
        # print("Length of tx_policies:",len(tx_policies))
        # print("Length of tx_weights:",len(tx_weights))
        # print("Length of link_indices:",len(link_indices))
        # print("Length of path_indices:",len(path_indices))
        # print("Length of sequ_indices:",len(sequ_indices))
        # print("Length of n_links:",len(n_links))
        # print("Length of n_paths:",len(n_paths))
        yield {"bandwith": avg_bw,
               "packets": pkts_gen,
               "tos":tos,
               "AvgPkS": AvgPkS,
               "link_capacity": link_capacities,
               "tx_policies":tx_policies,
               "tx_weights":tx_weights,
               "links": link_indices,
               "paths": path_indices,
               "sequences": sequ_indices,
               "n_links": n_links,
               "n_paths": n_paths}, delay


def transformation(x, y):
    """Apply a transformation over all the samples included in the dataset.

        Args:
            x (dict): predictor variable.
            y (array): target variable.

        Returns:
            x,y: The modified predictor/target variables.
        """
    return x, y


def input_fn(data_dir, transform=True, repeat=True, shuffle=False, generat_data=False):
    """This function uses the generator function in order to create a Tensorflow dataset

        Args:
            data_dir (string): Path of the data directory.
            transform (bool): If true, the data is transformed using the transformation function.
            repeat (bool): If true, the data is repeated. This means that, when all the data has been read,
                            the generator starts again.
            shuffle (bool): If true, the data is shuffled before being processed.

        Returns:
            tf.data.Dataset: Containing a tuple where the first value are the predictor variables and
                             the second one is the target variable.
        """
    ds = tf.data.Dataset.from_generator(lambda: generator(data_dir=data_dir, shuffle=shuffle,generat_data=generat_data),
                                        ({"bandwith": tf.float32,
                                          "packets": tf.float32,
                                          "tos": tf.float32,
                                          "AvgPkS": tf.float32,
                                          "link_capacity": tf.float32,
                                          "tx_policies": tf.float32,
                                          "tx_weights": tf.float32,
                                          "links": tf.int64,
                                          "paths": tf.int64,
                                          "sequences": tf.int64,
                                          "n_links": tf.int64, "n_paths": tf.int64},
                                        tf.float32),
                                        ({"bandwith": tf.TensorShape([None]),
                                          "packets": tf.TensorShape([None]),
                                          "tos": tf.TensorShape([None]),
                                          "AvgPkS": tf.TensorShape([None]),
                                          "link_capacity": tf.TensorShape([None]),
                                          "tx_policies": tf.TensorShape([None]),
                                          "tx_weights": tf.TensorShape([None]),
                                          "links": tf.TensorShape([None]),
                                          "paths": tf.TensorShape([None]),
                                          "sequences": tf.TensorShape([None]),
                                          "n_links": tf.TensorShape([]),
                                          "n_paths": tf.TensorShape([])},
                                         tf.TensorShape([None])))
    if transform:
        ds = ds.map(lambda x, y: transformation(x, y))

    if repeat:
        ds = ds.repeat()
    return ds






############### ProtoType 2 ########################################

# """
#    Copyright 2020 Universitat Politècnica de Catalunya

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# """

# import numpy as np
# import tensorflow as tf

# from datanetAPI import DatanetAPI

# POLICIES = np.array(['WFQ', 'SP', 'DRR'])
# WEIGHTS  = np.array(['90,5,5','80,10,10','75,25,5','70,20,10','65,25,10','60,30,10','50,40,10','33.3,33.3,33.3'])
# data_to_return = list()
# #when reading the evaluation data-set (i.e., generating the prediction) replace the line above by the line below!
# #WEIGHTS  = np.array(['90,5,5','80,10,10','75,25,5','70,20,10','65,25,10','60,30,10','50,40,10','33.3,33.3,33.4'])


# def policy(graph, node):
#     weight=-8
#     pol =   np.where(POLICIES==graph.nodes[node]['schedulingPolicy'])[0][0]
#     if (pol!=1):
#         weight=np.where(WEIGHTS==graph.nodes[node]['schedulingWeights'])[0][0]+1
#     return (pol, weight/8.)   #for 'SP' is always -1, otherwise between ]0.,1.]

# ################################ Custom Function ############################################
# def return_corresponding_data(data_dir, shuffle = False):
#         return data_to_return
# ########################################################################################################################################


# def generator(data_dir, shuffle = False):
#     """This function uses the provided API to read the data and returns
#        and returns the different selected features.

#     Args:
#         data_dir (string): Path of the data directory.
#         shuffle (string): If true, the data is shuffled before being processed.

#     Returns:
#         tuple: The first element contains a dictionary with the following keys:
#             - bandwith
#             - packets
#             - link_capacity
#             - links
#             - paths
#             - sequences
#             - n_links, n_paths
#             The second element contains the source-destination delay
#     """
#     tool = DatanetAPI(data_dir, [], shuffle)
#     ii=0
#     it = iter(tool)
#     for sample in it:
#         ###################
#         #  EXTRACT PATHS  #
#         ###################
#         routing = sample.get_routing_matrix()
#         traffic = sample.get_traffic_matrix()
#        # print("ROUTING",routing)
#        # print("TRAFFIC:", traffic)
#         nodes = len(routing)
#         #print("NODES: ",nodes)
#         # Remove diagonal from matrix
#         paths = routing[~np.eye(routing.shape[0], dtype=bool)].reshape(routing.shape[0], -1)
#         paths = paths.flatten()
       
#         #print("PATHS: ",paths)

#         ###################
#         #  EXTRACT LINKS  #
#         ###################
#         g = sample.get_topology_object()
     
#         cap_mat         = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)
#         tx_queue_mat    = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)
#         tx_weight_mat   = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)
        
#         for node in range(g.number_of_nodes()):
#             for adj in g[node]:
#                 cap_mat[node, adj] = g[node][adj][0]['bandwidth']*1.E-5  #BEWARE : SCALING  !
#                 tx_queue_mat[node, adj] = policy(g,node)[0]
#                 tx_weight_mat[node, adj] = policy(g,node)[1]

#         links = np.where(np.ravel(cap_mat) != None)[0].tolist()
#         link_capacities   = (np.ravel(cap_mat)[links]).tolist()
#         tx_policies       = (np.ravel(tx_queue_mat)[links]).tolist()
#         tx_weights         = (np.ravel(tx_weight_mat)[links]).tolist()
        
#         ids = list(range(len(links)))
#         links_id = dict(zip(links, ids))

#         path_ids = []
#         weight_ids = []
#         pol_ids = []
#         for path in paths:
#             new_path = []
#             for i in range(0, len(path) - 1):
#                 src = path[i]
#                 dst = path[i + 1]         
#                 new_path.append(links_id[src * nodes + dst])

#             path_ids.append(new_path)
#         ###################
#         #   MAKE INDICES  #
#         ###################
#         link_indices  = []
#         path_indices  = []
#         sequ_indices  = []
#         segment = 0
#         for i in range(len(path_ids)):
#             p=path_ids[i]
#             link_indices += p
#             path_indices += len(p) * [segment]
#             sequ_indices += list(range(len(p)))
#             segment += 1

#        # print("PATH LEVEL FEATURES:",len(link_indices),len(path_indices))
#         # Remove diagonal from matrix
#         traffic = traffic[~np.eye(traffic.shape[0], dtype=bool)].reshape(traffic.shape[0], -1)

#         result = sample.get_performance_matrix()
#         # Remove diagonal from matrix
#         result = result[~np.eye(result.shape[0], dtype=bool)].reshape(result.shape[0], -1)

#         avg_bw = []
#         pkts_gen = []
#         delay = []
#         tos = []
#         AvgPkS = []

#         ArrT = [] # Arrival Time
#         dur = []  # Duration
#         vol = []  # Volume
#         dL = [] # Deadline

# ##### Additional Features for deadline constraint problem for every Flow ######
#         lambda_value = 0.1  # Parameter for exponential distribution (controls mean arrival time)
#         lower_limit_duration = 5  # Lower limit for duration
#         upper_limit_duration = 30  # Upper limit for duration
#         lower_limit_volume = 100  # Lower limit for volume
#         upper_limit_volume = 1000  # Upper limit for volume


#         for i in range(result.shape[0]):
#             for j in range(result.shape[1]):
                
#                 flow = traffic[i, j]['Flows'][0]
#                 ###################################
#                 arrival_time = np.random.exponential(scale=1/lambda_value)
#                 ArrT.append(arrival_time)
#                 duration = np.random.uniform(lower_limit_duration, upper_limit_duration)
#                 dur.append(duration)
#                 volume = np.random.uniform(lower_limit_volume, upper_limit_volume)
#                 vol.append(volume)
#                 deadline = np.random.uniform(20,40) 
#                 dL.append(deadline)
#                 #####################################

#                 avg_bw.append(flow['AvgBw']*1.E-3)
#                 tos.append(float(flow['ToS']))
#                 AvgPkS.append(flow['SizeDistParams']['AvgPktSize']*1.E-4)
#                 pkts_gen.append(flow['PktsGen'])
#                 d = result[i, j]['AggInfo']['AvgDelay']
#                 delay.append(d)

#        # print("LINK LEVEL FEATURE SIZE:",len(avg_bw),len(pkts_gen),len(delay))

#         n_paths = len(path_ids)
#         n_links = max(max(path_ids)) + 1

#         for path,packs,abw in zip(paths,AvgPkS,avg_bw):
#           src = path[0]
#           dst = path[len(path)-1]
#           data_to_return.append([src,dst,path,g,packs,abw])
#          # print(ii,":","SOURCE:",src,"DESTINATION:",dst,"PATH:",path,"TOPOLOGY:",g,"Average Packets:",packs,"Average Bandwidth:",abw)
#           ii+=1
#  ######################################################   


#         yield {"bandwith": avg_bw,
#                "packets": pkts_gen,
#                "tos":tos,
#                "AvgPkS": AvgPkS,
#                "link_capacity": link_capacities,
#                "tx_policies":tx_policies,
#                "tx_weights":tx_weights,
#                "links": link_indices,
#                "paths": path_indices,
#                "sequences": sequ_indices,
#                "n_links": n_links,
#                "n_paths": n_paths,
#                "arrival_time":ArrT,
#                "duration": dur,
#                "volume": vol,
#                "deadline": dL
#              }, delay


# def transformation(x, y):
#     """Apply a transformation over all the samples included in the dataset.

#         Args:
#             x (dict): predictor variable.
#             y (array): target variable.

#         Returns:
#             x,y: The modified predictor/target variables.
#         """
#     return x, y


# def input_fn(data_dir, transform=True, repeat=True, shuffle=False):
#     """This function uses the generator function in order to create a Tensorflow dataset

#         Args:
#             data_dir (string): Path of the data directory.
#             transform (bool): If true, the data is transformed using the transformation function.
#             repeat (bool): If true, the data is repeated. This means that, when all the data has been read,
#                             the generator starts again.
#             shuffle (bool): If true, the data is shuffled before being processed.

#         Returns:
#             tf.data.Dataset: Containing a tuple where the first value are the predictor variables and
#                              the second one is the target variable.
#         """
#     ds = tf.data.Dataset.from_generator(lambda: generator(data_dir=data_dir, shuffle=shuffle),
#                                         ({"bandwith": tf.float32,
#                                           "packets": tf.float32,
#                                           "tos": tf.float32,
#                                           "AvgPkS": tf.float32,
#                                           "link_capacity": tf.float32,
#                                           "tx_policies": tf.float32,
#                                           "tx_weights": tf.float32,
#                                           "links": tf.int64,
#                                           "paths": tf.int64,
#                                           "sequences": tf.int64,
#                                           "n_links": tf.int64, "n_paths": tf.int64,
#                                           "arrival_time":tf.float32,
#                                           "duration":tf.float32,
#                                           "volume":tf.float32,
#                                           "deadline":tf.float32
#                                           },
#                                         tf.float32),
#                                         ({"bandwith": tf.TensorShape([None]),
#                                           "packets": tf.TensorShape([None]),
#                                           "tos": tf.TensorShape([None]),
#                                           "AvgPkS": tf.TensorShape([None]),
#                                           "link_capacity": tf.TensorShape([None]),
#                                           "tx_policies": tf.TensorShape([None]),
#                                           "tx_weights": tf.TensorShape([None]),
#                                           "links": tf.TensorShape([None]),
#                                           "paths": tf.TensorShape([None]),
#                                           "sequences": tf.TensorShape([None]),
#                                           "n_links": tf.TensorShape([]),
#                                           "n_paths": tf.TensorShape([]),
#                                           "arrival_time":tf.TensorShape([None]),
#                                           "duration":tf.TensorShape([None]),
#                                           "volume":tf.TensorShape([None]),
#                                           "deadline":tf.TensorShape([None])
#                                           },
#                                          tf.TensorShape([None])))
#     if transform:
#         ds = ds.map(lambda x, y: transformation(x, y))

#     if repeat:
#         ds = ds.repeat()
#     return ds
