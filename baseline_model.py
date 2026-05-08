import torch
import torch.nn as nn
import torchvision.models as models


class VolleyballBaselineModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=2):
        super(VolleyballBaselineModel, self).__init__()

        # 1. Feature Extractor: Pre-trained ResNet18
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # freeze the ResNet weights for the beginning (Transfer Learning)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        # 3. Classifier: Verbindet die LSTM-Ausgabe mit den 5 setting actions
        self.fc = nn.Linear(hidden_dim * 2, num_classes) # hidden_dim * 2 wegen Bidirectional

    def unfreeze_backbone(self):
        #Macht den ResNet-Teil trainierbar für Stage 2 Fine-Tuning
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x shape: (Batch, Frames, Channels, Height, Width)
        batch_size, seq_len, c, h, w = x.shape

        # prepare frames for CNN
        ii = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(ii)  # Output: (B*S, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # Zurück zu (B, S, 512)

        # LSTM processes the sequence
        lstm_out, _ = self.lstm(features)

        #take the output of the last step for classification
        last_time_step = lstm_out[:, -1, :]

        out = self.fc(last_time_step)
        return out


if __name__ == "__main__":
    # Test: create  a Fake-Video-Batch (2 Videos, 60 Frames)
    model = VolleyballBaselineModel(num_classes=5)
    fake_video = torch.randn(2, 60, 3, 224, 224)
    output = model(fake_video)
    print(f"Modell-Output Shape: {output.shape}")  # Erwartet: [2, 5]