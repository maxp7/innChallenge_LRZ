import numpy as np
import pandas as pd
import pandas_profiling as pp

#read Tables from node0 - node2
node_0 = pd.read_csv(r"Prepared Data/Data_Prepared_node0.csv",sep=";")
node_1 = pd.read_csv(r"Prepared Data/Data_Prepared_node1.csv",sep=";")
node_2 = pd.read_csv(r"Prepared Data/Data_Prepared_node2.csv",sep=";")

#create pandas profile for node 0
profile = pp.ProfileReport(node_0)
profile.to_file("Node-0-Profile.html")

#create pandas profile for node 1
profile = pp.ProfileReport(node_1)
profile.to_file("Node-1-Profile.html")

#create pandas profile for node 2
profile = pp.ProfileReport(node_2)
profile.to_file("Node-2-Profile.html")

