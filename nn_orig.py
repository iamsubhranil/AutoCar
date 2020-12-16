import random
import math
import numpy as np
import printmanager as pm

def sigmoid(values):
    #print("ip:", -values)
    #values = np.interp(values, (min(values), max(values)), (-4, 4))
    op = 1.0 / (1.0 + np.exp(-values))
    #print("op:", op)
    return op

def sigmoid_derivative(x, out):
    y = sigmoid(x)
    return y * (1.0 - y)

def tanh(x):
    return math.tanh(x)

def relu6(x):
    return min(max(0.0, x), 6.0)

def relu(values):
    return np.array([max(0.0, x) for x in values])

def relu_derivative(values, out):
    return np.array([1 if x > 0 else 0 for x in values])

def softplus(x):
    return math.log(1 + math.exp(x))

def softmax(x):
    print("softmax(", x,"):", end='')
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    ret = exps / np.sum(exps)
    print(ret)
    return ret

def softmax_derivative(S, out):
    S_vector = S.reshape(S.shape[0],1)
    S_matrix = np.tile(S_vector,S.shape[0])
    S_dir = np.diag(S) - (S_matrix * np.transpose(S_matrix))
    return S_dir

#def softmax_derivative(values, out):
#    return [0.01 * x for x in values]

class NeuralNetworkError(Exception):

    def __init__(self, message):
        super().__init__(message)

IS_MP_SUPPORTED = False
try:
    from multiprocessing import Pool, cpu_count
except ImportError:
    IS_MP_SUPPORTED = False

def process_level_worker(level, biases, output):
    res = []
    for nodeidx, node_weights in enumerate(level):
        o = 0
        for i, weight in enumerate(node_weights):
            o += output[i] * weight
        o += biases[nodeidx]
        res.append(sigmoid(o))
    return res

ACTIVAITON_DERIVATIVES = {
    sigmoid: sigmoid_derivative,
    relu: relu_derivative,
    softmax: softmax_derivative
}

class NeuralNetwork:

    LEARNING_RATE = 0.55

    def __init__(self, topo, randomize_weights=True, activation_funcs = None, weights=None, biases=None):
        #self.network = []
        self.weights = weights
        self.biases = biases
        topo2 = list(topo)
        self.topology = np.array(topo2)
        if activation_funcs is None:
            activation_funcs = [sigmoid] * (len(topo2) - 1)
            activation_funcs.append(sigmoid)
            #activation_funcs.append(softmax)
        self.activation_func = activation_funcs
        self.activation_derivative = [ACTIVAITON_DERIVATIVES[x] for x in self.activation_func]
        if self.biases is None:
            if randomize_weights:
                self.biases = [np.random.rand(x) for x in topo2]
                #print(self.biases)
            else:
                self.biases = [np.zeros(x) for x in topo2]

        if self.weights == None:
            topo2.append(0)
            if randomize_weights:
                self.weights = [np.random.rand(count, topo2[idx - 1]) *
                                          np.sqrt(2 / count) for idx, count in enumerate(topo2[:-1])]
            else:
                self.weights = [np.zeros((count, topo2[idx - 1])) for idx, count in enumerate(topo2[:-1])]
        #print("Initialization")
        #print("--------------")
        #self.dump()
        #print("--------------")

    def set_weights_and_biases(self, w, b):
        self.weights = w
        self.biases = b

    def dump(self, file=None):
        if file is None:
            for i, level in enumerate(self.weights):
                print("Level:", i + 1)
                for j, weight in enumerate(level):
                    print("\tNeuron:", j + 1, "bias:", self.biases[i][j], "weights:", weight)
        else:
            with open(file, "wb") as f:
                np.save(f, self.topology)
                np.save(f, self.weights)
                np.save(f, self.biases)

    def rgb(self, minimum, maximum, value):
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value-minimum) / (maximum - minimum)
        b = int(max(0, 255*(1 - ratio)))
        r = int(max(0, 255*(ratio - 1)))
        g = 255 - b - r
        return r, g, b

    def visualize(self):
        all_weight_sums = []
        prev_weight_sum = None
        for weight in self.weights[:0:-1]:
            # find sum of weights for each node in previous level
            if prev_weight_sum is None:
                weight_sums = np.sum(weight, axis=0)
            else:
                weight_sums = weight[0] * prev_weight_sum[0]
                for i, w in enumerate(weight[1:], 1):
                    weight_sums += w * prev_weight_sum[i]
            prev_weight_sum = weight_sums
            # append the sum for previous level
            all_weight_sums.insert(0, weight_sums)
        #print(all_weight_sums)
        from PIL import Image, ImageDraw
        from math import sqrt, floor
        for i, weight_sums in enumerate(all_weight_sums):
            # find the sqrt of length
            size = len(weight_sums)
            sq = sqrt(size)
            width = int(floor(sq))
            height = int(floor(sq))
            while width * height < size:
                height += 1
            # size of each color square
            sqr_width = 100
            sqr_height = 100
            level_image = Image.new("RGB", (width * sqr_width, height * sqr_height))
            draw_image = ImageDraw.Draw(level_image)
            present_height = 0
            present_width = 0
            putpixel_count = 0
            level_max = max(weight_sums)
            level_min = min(weight_sums)
            break_all = False
            for j in range(height):
                for k in range(width):
                    present_val = weight_sums[j * width + k]
                    #print(level_max, level_min, present_val)
                    x0, y0 = present_width, present_height
                    x1, y1 = x0 + sqr_width, y0 + sqr_height
                    draw_image.rectangle([x0, y0, x1, y1], fill=self.rgb(level_min, level_max, present_val))
                    present_width += sqr_width
                    putpixel_count += 1
                    if putpixel_count == size:
                        break_all = True
                        break
                if break_all:
                    break
                present_width = 0
                present_height += sqr_height
            level_image.save("level_%d.jpg" % i)

    @classmethod
    def load(cls, file):
        with open(file, "rb") as f:
            t = np.load(f, allow_pickle=True)
            w = np.load(f, allow_pickle=True)
            b = np.load(f, allow_pickle=True)
            nn = NeuralNetwork(t, False, None, w, b)
            return nn

    def process_input(self, input_value, return_all_outputs=False):
        output = np.array(input_value)
        if return_all_outputs:
            all_outputs = [output]
            net_outputs = [output]
        #lastlevel = len(level) - 1
        for levelidx, level in enumerate(self.weights[1:], 1):
            temp_output = np.dot(level, output) + self.biases[levelidx]
            #print(temp_output)
            new_output = self.activation_func[levelidx](temp_output)
            output = new_output
            if return_all_outputs:
                all_outputs.append(output)
                net_outputs.append(temp_output)
        #print(output)
        if return_all_outputs:
            return (all_outputs, net_outputs)
        else:
            return output

    def backprop(self, expected_output, generated_outputs, net_outputs, total_nabla_b,
                 total_nabla_w, len_weights, verbose=False):
        errors = None
        last_level = True
        for levelidx in range(1, len_weights, 1):
            #print(levelidx)
            levelidx = len_weights - levelidx
            derivative_outputs = self.activation_derivative[levelidx](net_outputs[levelidx], expected_output)
            #if verbose:
            #    print("level:", levelidx)
            #    print("net_outputs:", net_outputs[levelidx])
            #    print("generated_outputs:", generated_outputs[levelidx])
            #    print("derivative_outputs:", derivative_outputs)
            if last_level:
                # cost derivative
                dL = generated_outputs[-1] - expected_output
            else:
                #if verbose:
                #    print("errors:", errors)
                #    print("next_weights:", self.weights[levelidx + 1].T)
                dL = np.dot(self.weights[levelidx + 1].T, errors)
            #if verbose:
            #    print("dL:", dL)
            dL *= derivative_outputs
            #if verbose:
            #    print("dL *= derivative_outputs:", dL)
            prev_outputs = generated_outputs[levelidx - 1]
            #if verbose:
            #    print("prev_outputs:", prev_outputs)
            weight_delta = np.outer(prev_outputs, dL).T
            #if verbose:
            #    print("weight_delta = dot(dL, prev_outputs):", weight_delta)
            #    input()
            total_nabla_b[levelidx] += dL
            total_nabla_w[levelidx] += weight_delta
            errors = dL
            if last_level:
                last_level = False

    def train(self, input_values, output_values, verbose=False, epoch=30, mini_batch_count=100):
        #Neuron.LEARNING_RATE = 1 / len(input_values)
        if len(output_values[0]) != len(self.weights[-1]):
            raise NeuralNetworkError("Mismatched output size!")
        input_values = np.array(input_values)
        output_values = np.array(output_values)
        len_weights = len(self.weights)
        len_biases = len(self.biases)
        total_nabla_b = [0.0] * len_biases
        total_nabla_w = [0.0] * len_weights
        input_size = len(input_values)
        batch_length = input_size // mini_batch_count
        #pm.updateln("Starting epochs..")
        for e in range(epoch):
            #pm.updateln("Epoch %2d/%2d: " % (e + 1, epoch))
            # shuffle the training data
            #pm.status_add("Shuffling training data..")
            indices = np.arange(input_values.shape[0])
            np.random.shuffle(indices)
            input_values = input_values[indices]
            output_values = output_values[indices]
            #pm.status_update("Slicing into mini batches..")
            input_slices = [input_values[x:x+batch_length] for x in range(0, input_size, batch_length)]
            output_slices = [output_values[x:x+batch_length] for x in range(0, input_size, batch_length)]
            #if verbose:
            #   pm.status_update("Training: ")
            #   pm.status_add("Starting batches..")
            #else:
            #   pm.status_update("Training..")
            for slice_index, islice in enumerate(input_slices):
                #if verbose:
                #    pm.status_update("Running batch %3d/%3d (size:%4d).." % (slice_index, mini_batch_count, batch_length))
                for input_index, input_value in enumerate(islice):
                    generated_output, net_outputs = self.process_input(input_value, True)
                    self.backprop(output_slices[slice_index][input_index], generated_output,
                                    net_outputs, total_nabla_b, total_nabla_w, len_weights)
                #if verbose:
                #    pm.status_update("Updating weights for the batch..")
                avg = len(islice) # len(slice) for outside
                for idx in range(len_weights):
                    total_nabla_w[idx] *= NeuralNetwork.LEARNING_RATE / avg
                    self.weights[idx] -= total_nabla_w[idx]
                for idx in range(len_biases):
                    total_nabla_b[idx] *= NeuralNetwork.LEARNING_RATE / avg
                    self.biases[idx] -= total_nabla_b[idx]
            #if verbose:
            #    pm.status_pop()
            #pm.status_pop()
        #pm.println("Training done..", flush=True)

