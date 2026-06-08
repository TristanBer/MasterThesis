import torch
import torch.nn as nn


class VolleyballSlowFastModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True, dropout_p=0.5, alpha=4):
        """
        alpha: The frame rate ratio between the Fast and Slow pathways.
               If input is 32 frames and alpha=4, Fast gets 32 frames, Slow gets 8 frames.
        """
        super(VolleyballSlowFastModel, self).__init__()
        self.alpha = alpha

        # FIX: Load the pre-trained Kinetics-400 SlowFast model from PyTorchVideo via Torch Hub
        self.model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)

        # Freeze the backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # FIX: Access the head layer via PyTorchVideo's architectural naming ('blocks[5]')
        in_features = self.model.blocks[5].proj.in_features

        # Update the dropout probability inside the built-in head
        self.model.blocks[5].dropout = nn.Dropout(p=dropout_p)

        # Replace the final linear projection layer with your custom class size
        self.model.blocks[5].proj = nn.Linear(in_features, num_classes)

        # Ensure the updated head layers are completely unfrozen and trainable
        for param in self.model.blocks[5].parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreezes all layers for Stage 2 fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        x input shape from Dataloader: (B, T, C, H, W)
        Permuted to SlowFast expectation: (B, C, T, H, W)
        """
        # Strict dimension permutation fix to prevent channel/temporal mismatch crashes
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)

        # Fast pathway: sees all T frames
        fast_pathway = x

        # Slow pathway: temporally subsampled by factor of alpha
        slow_pathway = x[:, :, ::self.alpha, :, :]

        # PyTorchVideo SlowFast expects a multiplexed list of pathways: [slow_tensor, fast_tensor]
        return self.model([slow_pathway, fast_pathway])