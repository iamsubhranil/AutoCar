import numpy as np

def sigmoid(values):
    #print("ip:", -values)
    #values = np.interp(values, (min(values), max(values)), (-4, 4))
    op = 1.0 / (1.0 + np.exp(-values))
    #print("op:", op)
    return op

def sigmoid_derivative(x, out):
    y = sigmoid(x)
    return y * (1.0 - y)

#def softmax_derivative(values, out):
#    return [0.01 * x for x in values]

class NeuralNetworkError(Exception):

    def __init__(self, message):
        super().__init__(message)

ACTIVAITON_DERIVATIVES = {
    sigmoid: sigmoid_derivative
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

    def visualize(self, prefix):
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
        for i, weight_sums in enumerate(all_weight_sums):
            # find the sqrt of length
            size = len(weight_sums)
            sq = np.sqrt(size)
            width = int(np.floor(sq))
            height = int(np.floor(sq))
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
            level_image.save(prefix + ("level_%d.jpg" % i))

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
