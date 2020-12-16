import nn_orig as nn
import random

# crossover gene count denotes the number of parents which
# participate in generation of a new child
def crossover(parents, dim, num_children, crossover_gene_count=0):

    if crossover_gene_count < 2:
        crossover_gene_count = len(parents)
    children = []

    idx_ranges = [dim[i-1] * dim[i] for i in range(1, len(dim))]
    #print(idx_ranges)
    total_nodes = sum(idx_ranges)
    for i, _ in enumerate(idx_ranges[1:], 1):
        idx_ranges[i] += idx_ranges[i-1]
    idx_ranges.insert(0, 0)
    #print(idx_ranges)
    for _ in range(num_children):
        # sample pivot points for each parent
        # choose 1 less point, use to the last parent
        # to fill the rest of it
        crossover_parents = random.sample(parents, k=crossover_gene_count)
        points = sorted(random.sample(range(1, total_nodes), k=crossover_gene_count-1))
        points.append(total_nodes)
        last_layer, last_node, last_weight = 0, 0, 0
        child = nn.NeuralNetwork(dim, False)
        for point, parent in zip(points, crossover_parents):
            if point == total_nodes:
                layer = len(dim) - 1
            else:
                layer = 0
                while point >= idx_ranges[layer]:
                    layer += 1
            abs_idx = point - idx_ranges[layer - 1]
            node = abs_idx // dim[layer - 1]
            weight = abs_idx % dim[layer - 1]
            # print("point:", point, "layer:", layer, "node:", node, "weight:", weight)
            # copy the remaining weights of the node first
            rem = parent.weights[last_layer][last_node][last_weight:]
            if len(rem) > 0:
                child.weights[last_layer][last_node][last_weight:] = rem
                child_frac = (last_weight / dim[last_layer - 1])
                parent_frac = 1 - child_frac
                child.biases[last_layer][last_node] += parent_frac * parent.biases[last_layer][last_node]
            last_weight = 0
            last_node += 1
            # check if we are in a different layer
            if last_layer < layer:
                # copy the remaining nodes of the layer
                while last_layer < layer:
                    child.weights[last_layer][last_node:] = parent.weights[last_layer][last_node:]
                    child.biases[last_layer][last_node:] = parent.biases[last_layer][last_node:]
                    last_layer += 1
                    last_node = 0
            child.weights[last_layer][last_node:node] = parent.weights[last_layer][last_node:node]
            child.biases[last_layer][last_node:node] = parent.biases[last_layer][last_node:node]
            last_node = node
            if weight > 0:
                child.weights[last_layer][last_node][:weight] = \
                        parent.weights[last_layer][last_node][:weight]
                parent_frac = (weight / dim[last_layer - 1])
                child.biases[last_layer][last_node] = parent_frac * parent.biases[last_layer][last_node]

            last_layer, last_node, last_weight = layer, node, weight

        children.append(child)

    #print("Parents:")
    #for parent in parents:
    #    print("Parent:")
    #    parent.dump()

    #print("\n\nChildren:")
    #for child in children:
    #    print("Child:")
    #    child.dump()

    return children

def mutation(children, mutation_probability=0.1):
    dim = children[0].topology
    num_layers = len(dim)
    for child in children:
        if random.uniform(0.0, 1.0) <= mutation_probability:
            layer = random.randint(1, num_layers - 1)
            node = random.randint(0, dim[layer] - 1)
            weight = random.randint(0, dim[layer - 1] - 1)
            child.weights[layer][node][weight] = random.random()

# inputs are carais, outputs are the best NNs
# n best candidates are selected
def selection(carais, n=5):
    best = sorted(carais, key=lambda x: x.score)
    print([x.score for x in best[-n:]])
    # best networks will have the highest scores
    return [ai.network for ai in best[-n:]]

if __name__ == "__main__":
    generation()

