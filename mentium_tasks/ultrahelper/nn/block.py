import torch
from ultralytics.nn.modules.block import C2f
from .register import register_module


@register_module('base')
@register_module('repeat')
class TracableC2f(C2f):
    """
    Equivalent implementation to C2f but symbolically tracables.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Basically need to re-write this from origianl C2f class:

        def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''''Forward pass through C2f layer.'''
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))
        """
        # Double channels (e.g., 64 → 128)
        x_expanded = self.cv1(x)

        # 2 - number of chunks
        # 1 - dimension to split along
        x_split = x_expanded.chunk(2, 1)

        # We know the number of splits is always 2 for C2f
        outputs = [x_split[0], x_split[1]]

        # Process through bottleneck modules with explicit loop
        # Each bottleneck takes the previous output and produces a new feature map
        prev = x_split[1]
        for module in self.m:
            prev = module(prev)
            outputs.append(prev)

        return self.cv2(torch.cat(outputs, 1))
