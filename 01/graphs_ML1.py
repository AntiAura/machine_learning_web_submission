import numpy as np

iterations = 1000 #number of page rank iterations

size = 200
graph = np.zeros((size, size))
p = 0.05

#generate edges using Erdösz-Renyi
for i in range(size):
	for j in range(i):
		rand = np.random.rand()
		if rand <= p:
			graph[i,j] = 1
			graph[j,i] = 1

row_sum = np.sum(graph, axis=1)
row_sum[row_sum == 0] = 1
graph = graph / row_sum #normalize

r = np.ones((size,1)) / size
A = graph.copy()
B = graph.copy()
#print(A)

for i in range(iterations):
	B = np.matmul(A,B)
B = np.matmul(B,r)

page_rank_1 = B.copy()

print("page_rank Erdösz Renyi:")
#print(B)


graph = np.zeros((size, size))
m = 5

graph[0:m, 0:m] = 1 #start with a small fully connected graph

#introduce new vertices / generate edges using Erdösz-Renyi
for i in range(5,size):
	for j in range(i):
		prob = np.sum(graph[j]) / np.sum(graph)
		rand = np.random.rand()
		if rand <= prob:
			graph[i,j] = 1
			graph[j,i] = 1

#print(graph)

row_sum = np.sum(graph, axis=1)
row_sum[row_sum == 0] = 1
graph = graph / row_sum #normalize

r = np.ones((size,1)) / size
A = graph.copy()
B = graph.copy()
#print(A)

for i in range(iterations):
	B = np.matmul(A,B)
B = np.matmul(B,r)

page_rank_2 = B.copy()

print("page_rank Barabasi Albert:")
#print(B)