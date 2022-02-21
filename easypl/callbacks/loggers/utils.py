class ImageCollector(object):
    def __init__(
            self,
            mode='first',
            max_images=1,
            score_func=None,
            largest=True,
            dataset_size=None
    ):
        self.mode = mode
        self.max_images = max_images
        self.score_func = score_func
        self.largest = largest
        self.dataset_size = dataset_size

        self.idx = 0

    def update(self, output, target):
        ...

    def compute(self):
        ...

    def reset(self):
        self.idx = 0
        ...
