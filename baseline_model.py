import torch
import torch.nn as nn
import torchvision.models as models


class VolleyballBaselineModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=2, dropout_p=0.5):
        super(VolleyballBaselineModel, self).__init__()

        # 1. Feature Extractor: Pre-trained ResNet18
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze ResNet weights for Stage 1 (Transfer Learning)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 2. Temporal Analysis: Bi-Directional LSTM
        # ResNet18 outputs 512 features per frame
        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        # 3. Classifier: Dropout + Linear
        # hidden_dim * 2 because bidirectional; dropout regularises the LSTM output
        # before the decision layer, directly targeting the overfitting
        self.fc = nn.Sequential(nn.Dropout(p=dropout_p),nn.Linear(hidden_dim * 2, num_classes))

    def unfreeze_backbone(self):
        """Call this to switch from Stage 1 to Stage 2 training."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x shape: (Batch, Frames, Channels, Height, Width)
        batch_size, seq_len, c, h, w = x.shape

        # Flatten all frames into the batch dimension for the CNN
        ii = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(ii)   # (B*S, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (B, S, 512)

        # LSTM processes the full frame sequence
        lstm_out, _ = self.lstm(features)

        # Only the last time-step feeds the classifier
        last_time_step = lstm_out[:, -1, :]

        out = self.fc(last_time_step)
        return out


if __name__ == "__main__":
    model = VolleyballBaselineModel(num_classes=5)
    fake_video = torch.randn(2, 60, 3, 224, 224)
    output = model(fake_video)
    print(f"Output shape: {output.shape}")   # Expected: [2, 5]