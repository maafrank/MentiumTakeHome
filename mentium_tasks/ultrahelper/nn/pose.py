import torch
from ultralytics.nn.modules.head import Pose, Detect
from .register import register_module
import torch.nn as nn

@register_module('head')
class ModifiedPose(Pose):
    """
    Modify Pose module without changing its architecture. 
    Implement 'forward_head' and 'forward_postprocessor' methods
    """
    def get_head(self,)->'ModifiedPoseHead':
        return ModifiedPoseHead(self)
    def get_postprocessor(self,)->'ModifiedPosePostprocessor':
        return ModifiedPosePostprocessor(self)
    def forward_head(self,x):
        """
        Contains maximum amount of operations excluding 'view' methods 
        
        Hardware-compatible operations (4D tensors only).
        Runs all convolutions and returns raw 4D feature maps.

        Args:
            x: List of input tensors from backbone [P3, P4, P5]
               Each tensor is [B, C, H, W]

        Returns:
            List of tuples: [(box_feat, cls_feat, kpt_feat), ...]
            All tensors are 4D: [B, C, H, W]
        """
        outputs = []
        for i in range(self.nl):
            # Box regression features (from Detect)
            box_feat = self.cv2[i](x[i])  # [B, C_box, H, W]

            # Classification features (from Detect)
            cls_feat = self.cv3[i](x[i])  # [B, C_cls, H, W]

            # Keypoint features (from Pose)
            kpt_feat = self.cv4[i](x[i])  # [B, C_kpt, H, W]

            outputs.append((box_feat, cls_feat, kpt_feat))

        return outputs
    def forward_postprocessor(self, feats):
        """
        Contains everything else left (including 'view' methods)

        CPU operations (view/reshape/decode operations).
        Takes 4D feature maps and applies all reshape and decoding logic.

        Args:
            feats: List of tuples from forward_head
                   [(box_feat, cls_feat, kpt_feat), ...]

        Returns:
            Same format as original Pose.forward()
        """
        # Get batch size from first feature
        bs = feats[0][0].shape[0]

        # Concatenate box and class features into x_list
        x_list = []
        for box_feat, cls_feat, kpt_feat in feats:
            x_list.append(torch.cat([box_feat, cls_feat], 1))

        # Concatenate keypoints
        kpt_list = []
        for box_feat, cls_feat, kpt_feat in feats:
            kpt_list.append(kpt_feat.view(bs, self.nk, -1))
        kpt = torch.cat(kpt_list, 2)

        # Training mode: return raw features
        if self.training:
            return x_list, kpt

        # Inference/Validation mode: decode predictions
        from ultralytics.nn.modules.head import Detect
        y = Detect._inference(self, x_list)
        pred_kpt = self.kpts_decode(bs, kpt)

        # Return decoded predictions
        # If not exporting, also return raw features (x_list, kpt) for validation loss
        return torch.cat([y, pred_kpt], 1) if self.export else (torch.cat([y, pred_kpt], 1), (x_list, kpt))

    def forward(self, x):
        """
        Full forward pass (for training/testing without split).
        """
        feats = self.forward_head(x)
        preds = self.forward_postprocessor(feats)
        return preds
    
class ModifiedPoseHead(ModifiedPose):
    def __init__(self,mpose:ModifiedPose):
        nn.Module.__init__(self,)
        self.__dict__.update(mpose.__dict__)
    def forward(self,x):
        with torch.no_grad():
            return self.forward_head(x)

class ModifiedPosePostprocessor(ModifiedPose):
    def __init__(self,mpose:ModifiedPose):
        nn.Module.__init__(self,)
        self.__dict__.update(mpose.__dict__)
    def forward(self,feats):
        with torch.no_grad():
            return self.forward_postprocessor(feats)