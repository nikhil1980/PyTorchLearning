# from torchvision import models
import torch
import numpy as np


# =============================== #
#      Initializing Tensor
# =============================== #
def basic_initialization():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Underlying device for this code: {device}")
    zero = torch.zeros(3)
    print(f"{zero}")
    one = torch.ones(3)
    print(f"{one}")
    print(f"Type: {type(one)}")
    my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64,
                             device=device, requires_grad=True)
    print(my_tensor, my_tensor.dtype, my_tensor.device, my_tensor.shape, my_tensor.requires_grad)

    """ Other Initialization methods for Tensor"""
    x = torch.empty(size=(3, 3))
    print(x)
    y = torch.ones((3, 3))
    print(f"All ones: {y}")
    z = torch.rand((3, 3))
    print(f"Random Matrix {z}")
    x1 = torch.eye(4, 4)  # Identity Matrix
    print(f"Identity Matrix: {x1}")
    y1 = torch.arange(start=0, end=10, step=2)
    print(y1)

    z1 = torch.linspace(start=0.1, end=1, steps=10)  # x = [0.1, 0.2, ..., 1]
    print(f"Linear Spacing: {z1}")
    x11 = torch.empty(size=(1, 5)).normal_(mean=0, std=1)  # Standard Normal distribution with mean=0, std=1
    print(f"Standard Normal Distribution: {x11}")

    y11 = torch.empty(size=(2, 5)).uniform_(0)  # Values from a uniform distribution low=0, high=1
    print(f"Uniform Distribution: {y11}")

    a = torch.diag(torch.ones(3))  # Diagonal matrix of shape 3x3
    print(f"Diagonal or Identity Matrix of shape 3x3: {a}")

    # How to make initialized tensors to other types (int, float, double)
    # These will work even if you're on CPU or CUDA!
    tensor = torch.arange(4)  # [0, 1, 2, 3] Initialized as int64 by default
    print(f"Original Tensor: {tensor}\n")
    print(f"Converted Boolean: {tensor.bool()}")  # Converted to Boolean: 1 if nonzero
    print(f"Converted int16 {tensor.short()}")  # Converted to int16
    print(f"Converted int64 {tensor.long()}")  # Converted to int64 (This one is very important, used super often)
    print(f"Converted float16 {tensor.half()}")  # Converted to float16
    print(f"Converted float32 {tensor.float()}")  # Converted to float32 (This one is very important, used super often)
    print(f"Converted float64 {tensor.double()}")  # Converted to float64

    # Array to Tensor conversion and vice-versa
    np_array = np.zeros((5, 5))
    print(f"Original Numpy array: {np_array}")
    tensor = torch.from_numpy(np_array)
    np_array_again = (tensor.numpy())  # np_array_again will be same as np_array (perhaps with numerical round offs)
    print(f"Tensor to Numpy converted array: {np_array_again}")


def basic_math_comparison():
    # =============================================================================== #
    #                        Tensor Math & Comparison Operations                      #
    # =============================================================================== #

    x = torch.tensor([1, 2, 3])
    print(f"x: {x}")
    y = torch.tensor([9, 8, 7])
    print(f"y: {y}")

    # -- Addition --
    z1 = torch.empty(3)
    torch.add(x, y, out=z1)  # This is one way
    z2 = torch.add(x, y)  # This is another way
    z = x + y  # This is my preferred way, simple and clean.
    print(f"Addition of x and y is z: {z}")

    # -- Subtraction --
    z = x - y  # We can do similarly as the preferred way of addition
    print(f"Subtraction of {x} from {y} is: {z}")

    # -- Division (A bit clunky) --
    z = torch.true_divide(x, y)  # Will do element wise division if of equal shape
    print(f"Element wise division of x with y if both of equal shape")
    print(f"z: {z}")

    # -- Inplace Operations --
    t = torch.zeros(3)

    t.add_(x)  # Whenever we have operation followed by _ it will mutate the tensor in place
    t += x  # Also inplace: t = t + x is not inplace, so a bit confusing.

    # -- Exponentiation (Element wise if vector or matrices) --
    z = x.pow(2)  # z = [1, 4, 9]
    print(f"{x} raised to power 2: {z}")
    z = x ** 2  # z = [1, 4, 9]

    # -- Simple Comparison --
    z = x > 0  # Returns [True, True, True]
    z = x < 0  # Returns [False, False, False]

    # -- Matrix Multiplication --
    x1 = torch.rand((2, 5))
    x2 = torch.rand((5, 3))
    x3 = torch.mm(x1, x2)  # Matrix multiplication of x1 and x2, out shape: 2x3
    print(f"{x1} multipy by {x2} is :{x3}")
    x3 = x1.mm(x2)  # Similar as line above

    # -- Matrix Exponentiation --
    matrix_exp = torch.rand(5, 5)
    # matrix_exp = matrix_exp.mm(matrix_exp).mm(matrix_exp)
    print(f"Matrix Exponentiation: {matrix_exp.matrix_power(3)}")

    # -- Element wise Multiplication --
    z = x * y  # z = [9, 16, 21] = [1*9, 2*8, 3*7]
    print(f"Element wise multiplication of {x} and {y}: {z}")

    # -- Dot product --
    z = torch.dot(x, y)  # Dot product, in this case z = 1*9 + 2*8 + 3*7
    print(f"Dot product of {x} and {y} is a scalar: {z}")

    # -- Batch Matrix Multiplication --
    batch = 32
    n = 10
    m = 20
    p = 30
    tensor1 = torch.rand((batch, n, m))
    tensor2 = torch.rand((batch, m, p))
    out_bmm = torch.bmm(tensor1, tensor2)  # Will be shape: (b x n x p)
    print(f"Batch {b} multiplication of {tensor1} and {tensor2}: {out_bmm}")

    # -- Example of broadcasting --
    x1 = torch.rand((5, 5))
    x2 = torch.ones((1, 5))
    z = (x1 - x2)  # Shape of z is 5x5: How? The 1x5 vector (x2) is subtracted for each row in the 5x5 (x1)
    z = (x1 ** x2)  # Shape of z is 5x5: How? Broadcasting! Element wise exponentiation for every row

    # Other useful tensor operations
    sum_x = torch.sum(x, dim=0)  # Sum of x across dim=0 (which is the only dim in our case), sum_x = 6
    print(f"Sum of {x} across row (dim=0): {sum_x}")

    values, indices = torch.max(x, dim=0)  # Can also do x.max(dim=0)
    print(f"Max of {x} across row (dim=0): {values} at index: {indices}")

    values, indices = x.min(dim=0)  # Can also do x.min(dim=0)
    print(f"Min of {x} across row (dim=0): {values} at index: {indices}")

    abs_x = torch.abs(x)  # Returns x where abs function has been applied to every element

    z = torch.argmax(x, dim=0)  # Gets index of the maximum value
    print(f"Argmax (index of max element) of {x} across row (dim=0): {z}")

    z = torch.argmin(x, dim=0)  # Gets index of the minimum value
    print(f"Argmin (index of min element) of {x} across row (dim=0): {z}")

    mean_y = torch.mean(y.float(), dim=0)  # mean requires x to be float
    print(f"Mean of {y}: {mean_y}")

    z = torch.eq(x, y)  # Element wise comparison, in this case z = [False, False, False]
    sorted_y, indices = torch.sort(y, dim=0, descending=False)
    print(f"Sorted {y} in ascending order: {sorted_y} with indices: {indices}")

    z = torch.clamp(x, min=0)
    # All values < 0 set to 0 and values > 0 unchanged (this is exactly ReLU function)
    # If you want to values over max_val to be clamped, do torch.clamp(x, min=min_val, max=max_val)
    z = torch.clamp(x, min=2, max=2)
    print(f"All values of {x} < {2} set to {2} and values > {2} set to {2} [Modified Relu]: {z}")

    x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  # True/False values
    z = torch.any(x)  # will return True, can also do x.any() instead of torch.any(x)
    z = torch.all(x)  # will return False (since not all are True), can also do x.all() instead of torch.all()


def tensor_indexing():
    # ============================================================= #
    #                        Tensor Indexing                        #
    # ============================================================= #
    batch_size = 10  # Number of samples
    features = 25  # Number of features in each sample

    x = torch.rand((batch_size, features))

    # Get first shapes
    print(f"Number of rows (samples): {x.shape[0]} or {x.size(dim=0)}")  # Number of rows
    print(f"Number of coloumns (): {x.shape[1]} or {x.size(dim=1)}")  # Number of coloumns

    # Get first sample's all 25 features
    print(x[0].shape)  # shape [25], this is same as doing x[0,:]

    # Get the first feature for all samples
    print(x[:, 0].shape)  # shape [10]

    # For example: Want to access third sample in the batch and the first ten features
    print(x[2, 0:10].shape)  # shape: [10]

    # For example, we can use this to, assign certain elements
    x[0, 0] = 100

    # Fancy Indexing
    x = torch.arange(10)
    print(f"x: {x}")

    indices = [2, 5, 8]
    print(x[indices])  # x[indices] = [2, 5, 8]

    x = torch.rand((3, 5))
    print(f"x: {x}")
    row_vector = torch.tensor([1, 0])
    col_vector = torch.tensor([4, 0])
    # Gets a tensor of element @ row_2, col_5 and row_1, col_1
    print(f"tensor of element @ row_2, col_5 and row_1, col_1: {x[row_vector, col_vector]}")

    # More advanced indexing
    x = torch.arange(10)
    print(x[(x < 2) | (x > 8)])  # will be [0, 1, 9]
    print(x[x.remainder(2) == 0])  # will be [0, 2, 4, 6, 8]

    # Useful operations for indexing

    # gives [0, 2, 4, 6, 8, 10, 6, 7, 8, 9], all values x > 5 yield x, else x*2
    print(f"All values x > 5 yield x, else x*2: {torch.where(x > 5, x, x * 2)}")

    x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()  # x = [0, 1, 2, 3, 4]

    # The number of dimensions, in this case 1. if x.shape is 5x5x5 ndim would be 3
    print(f"Number of dimensions of {x}: {x.ndimension()}")

    # The number of elements in x (in this case it's trivial because it's just a vector)
    x = torch.arange(10)
    print(f"Number of elements in {x}: {x.numel()}")


def tensor_reshaping():
    # ============================================================= #
    #                        Tensor Reshaping                       #
    # ============================================================= #

    x = torch.arange(9)

    # Let's say we want to reshape it to be 3x3
    x_3x3 = x.view(3, 3)
    print(f"Original x: {x}")
    print(f"Reshaped x (3x3) via view(): {x_3x3}")

    # We can also do (view and reshape are very similar)
    # and the differences are in simple terms (I'm no expert at this),
    # is that view acts on contiguous tensors meaning if the
    # tensor is stored contiguously in memory or not, whereas
    # for reshape it doesn't matter because it will copy the
    # tensor to make it contiguously stored, which might come
    # with some performance loss.
    x_3x3 = x.reshape(3, 3)
    print(f"Reshaped x (3x3) via reshape(): {x_3x3}")

    # If we for example do:
    y = x_3x3.t()
    print(f"Transpose of {x_3x3}: {y}")

    # This will return False and if we try to use view now, it won't work!
    # This is because in memory it was stored [0, 1, 2, ... 8], whereas now it's [0, 3, 6, 1, 4, 7, 2, 5, 8]
    # The jump is no longer 1 in memory for one element jump (matrices are stored as a contiguous block, and
    # using pointers to construct these matrices). This is a bit complicated and I need to explore this more
    # as well, at least you know it's a problem to be cautious of! A solution is to do the following
    print(f"This will return as {y.is_contiguous()} because y is split in two 2D and "
          f"if we try to use view now, it won't work!")
    # y.view(9) would cause an error, reshape however won't
    print(f"This will work: {y.reshape(9)}")

    print(y.contiguous().view(9))  # Calling .contiguous() before view and it works

    # Moving on to another operation, let's say we want to add two tensors dimensions
    x1 = torch.rand(2, 5)
    x2 = torch.rand(2, 5)

    # Shape: 4x5
    print(f"Concatenating {x1} \n and {x2}\n{torch.cat((x1, x2), dim=0)} with shape {torch.cat((x1, x2), dim=0).shape}")
    # Shape 2x10
    print(f"Concatenating {x1} \n and {x2}\n{torch.cat((x1, x2), dim=1)} with shape {torch.cat((x1, x2), dim=1).shape}")

    # Let's say we want to unroll x1 into one long vector with 10 elements, we can do:
    z = x1.view(-1)  # And -1 will unroll everything
    print(f"Flatten {x1} into a vector: {z}")

    # If we instead have an additional dimension, and we wish to keep those as is we can do:
    batch = 64
    x = torch.rand((batch, 2, 5))
    z = x.view(batch, -1)  # And z.shape would be 64x10, this is very useful stuff and is used all the time
    print(f"Flatten batch of {batch} {x} into a vector: {z} with shape: {z.shape}")

    # Let's say we want to switch x-axis so that instead of 64x2x5 we have 64x5x2
    # i.e., we want dimension 0 to stay, dimension 1 to become dimension 2, dimension 2 to become dimension 1
    # Basically you tell permute where you want the new dimensions to be, torch.transpose is a special case
    # of permute (why?)
    z = x.permute(0, 2, 1)

    # Splits x last dimension into chunks of 2 (since 5 is not integer div by 2) the last dimension
    # will be smaller, so it will split it into two tensors: 64x2x3 and 64x2x2
    z = torch.chunk(x, chunks=2, dim=1)
    print(z[0].shape)
    print(z[1].shape)

    # Let's say we want to add a dimension
    x = torch.arange(10)  # Shape is [10], let's say we want to add another dimension, so we have 1x10
    print(f"We make a flat tensor {x} to a row-vector: {x.unsqueeze(0)} with shape: {x.unsqueeze(0).shape}")  # 1x10
    print(f"We make a flat tensor {x} to a col-vector: {x.unsqueeze(1)} with shape: {x.unsqueeze(1).shape}")  # 10x1

    # Let's say we have x which is 1x1x10, and we want to remove a dim, so we have 1x10
    x = torch.arange(10).unsqueeze(0).unsqueeze(1)
    print(x.shape)

    # Perhaps unsurprisingly
    z = x.squeeze(1)  # can also do .squeeze(0) both returns 1x10


if __name__ == '__main__':
    basic_initialization()

    #basic_math_comparison()

    #tensor_indexing()

    #tensor_reshaping()
