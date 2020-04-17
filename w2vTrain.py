from gensim.models import Word2Vec
import networkx as nx
import os
import logging
import json
import pickle


basedir = '/mnt/3A0F13EC43ACDB83/workspace/dblp-ref/'
outputdir = '/mnt/3A0F13EC43ACDB83/workspace/dblp_outputs/'

# dblpfile= open('/mnt/3A0F13EC43ACDB83/workspace/dblp-ref/dblp-ref-0.json')
# titles = dblpfile.readlines()
# dblpfile.close()
# for x in range(10):
#     doc = json.loads(titles[x])
#     print(doc["id"])

title_list = []
list_for_graph = []
dg = nx.DiGraph()
page_ranks = {}

# parse data file and create list
def parseAllJson(file_path):
   f = open(file_path)
   lines = f.readlines()
   f.close()
   for line in lines:
       doc = json.loads(line)
       title_list.append(doc.get("title"))
       list_for_graph.append([doc.get("id"),doc.get("references")])
# create directed graph    
def create_graph(data_list):
    for item in data_list:
        node = item[0]
        edges = item[1]
        dg.add_node(node)
        if edges != None:
            for edge in edges:
                dg.add_edge(node,edge)



with os.scandir(basedir) as files_in_dir:
    for doc_file in files_in_dir:
        parseAllJson(doc_file)
    with open(outputdir+'dblpv10_titles', 'w+') as f:
        for item in title_list:
            f.write(item+"\n")
    # with open(outputdir+'graph_data','w+') as f:
    #     f.writelines(list_for_graph)
    # for i in range(10):
    #     print(list_for_graph[i][0])
    # create_graph(list_for_graph)
    # nx.write_gpickle(dg,outputdir+'paper_graph.gpickle')
    
    # dg = nx.read_gpickle(outputdir+'paper_graph.gpickle')
    # print('read graph into memory')
    # with open(outputdir+'paper_graph.gpickle','r') as f:
    #     # dg = nx.read_gpickle(f)
    #     print('page ranking..')
    #     page_ranks = nx.pagerank(dg, alpha=0.8, max_iter=100, weight=1)
    #     print(page_ranks)





print("total title",len(title_list))
