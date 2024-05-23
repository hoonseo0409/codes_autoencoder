import numpy as np

class Group_finder():
    def __init__(self, adjacency_matrix):
        assert(adjacency_matrix.shape[0] == adjacency_matrix.shape[1] and len(adjacency_matrix.shape) == 2)
        self.adjacency_matrix = adjacency_matrix
        self.number_of_vertices = adjacency_matrix.shape[0]
    
    def DFSUtil(self, temp, v_from, visited):
        visited[v_from] = 1
        temp.append(v_from)
        for v_to in range(self.number_of_vertices):
            if visited[v_to] == 0 and self.adjacency_matrix[v_from, v_to] == 1:
                temp = self.DFSUtil(temp, v_to, visited)
        return temp
    
    def connected_components(self):
        visited = np.zeros(self.number_of_vertices)
        cc = []
        for v_from in range(self.number_of_vertices):
            if visited[v_from] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v_from, visited))
        return cc

# test_adjacency_matrix = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 1, 0]])
# test_adjacency_matrix = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 1, 0, 1, 1]])
# print(test_adjacency_matrix)
# group_finder_inst = Group_finder(test_adjacency_matrix)
# print(group_finder_inst.connected_components())