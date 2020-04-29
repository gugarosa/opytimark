import opytimark.utils.decorator as d

class Benchmark:
    """
    """

    def __init__(self, name='Benchmark', dims=1, continuous=False, convex=False, differentiable=False, multimodal=False, separable=False):
        """Initialization method.

        Args:
        

        """

        #
        self.name = name

        #
        self.dims = dims

        #
        self.continuous = continuous

        #
        self.convex = convex

        #
        self.differentiable = differentiable

        #
        self.multimodal = multimodal

        #
        self.separable = separable

    @d.check_dimension
    def __call__(self, x):
        """
        """

        return x