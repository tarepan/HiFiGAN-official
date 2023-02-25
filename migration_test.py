import torch


def get_padding(kernel_size, dilation):
    """equal to `int(d * (k-1) / 2)`."""
    return int((kernel_size*dilation - dilation)/2)

def test_padding():
    # Kernel is restricted to odd number
    for k, d in [(3, 1), (5, 1), (3, 2), (5, 2), (3, 3), (5, 3)]: # (4, 1) will fail.
        DirectNet = torch.nn.Conv1d(1, 1, k, dilation=d, padding=get_padding(k, d))
        SameNet   = torch.nn.Conv1d(1, 1, k, dilation=d, padding="same")
        i = torch.tensor([[1., 2., 3., 4., 5.,]])
        o_direct = DirectNet(i)
        o_same = SameNet(i)
        print(i.shape, o_direct.shape, o_same.shape)
        assert i.shape == o_direct.shape, "Direct Padding change tensor shape."
        assert i.shape == o_same.shape,   "Same Padding change tensor shape."
        assert o_direct.shape == o_same.shape, "Something wrong."
    print("test passed.")

test_padding()
