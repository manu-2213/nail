# Implementation of the Autograd like backprop

This is based on the implementation from Andrej Karpathy. This first implementation is done at the value level. A next implementation will use tensors / ndarrays to vectorize operations. 

This implementation does not use a symbolic approach to compute derivatives. Instead, it uses a numerical approach. Both implementations should be tested in the future.