import torch
import torch.nn as nn
import torchvision.models as models


class VolleyballBaselineModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=2):
        super(VolleyballBaselineModel, self).__init__()

        # 1. Feature Extractor: Pre-trained ResNet18
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        # Wir entfernen die letzte Klassifizierungsschicht (fc)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Wir frieren die ResNet-Gewichte für den Anfang ein (Transfer Learning)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 2. Zeitliche Analyse: Bi-Directional LSTM
        # ResNet18 gibt 512 Features aus
        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        # 3. Classifier: Verbindet die LSTM-Ausgabe mit deinen 5 Klassen
        # hidden_dim * 2 wegen Bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Frames, Channels, Height, Width)
        batch_size, seq_len, c, h, w = x.shape

        # Alle Frames des Batches flachklopfen für das CNN
        ii = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(ii)  # Output: (B*S, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # Zurück zu (B, S, 512)

        # LSTM verarbeitet die Sequenz
        lstm_out, _ = self.lstm(features)

        # Wir nehmen nur den Output des letzten Zeitschritts für die Klassifizierung
        last_time_step = lstm_out[:, -1, :]

        out = self.fc(last_time_step)
        return out


if __name__ == "__main__":
    # Kurzer Test: Erstelle ein Fake-Video-Batch (2 Videos, 60 Frames)
    model = VolleyballBaselineModel(num_classes=5)
    fake_video = torch.randn(2, 60, 3, 224, 224)
    output = model(fake_video)
    print(f"Modell-Output Shape: {output.shape}")  # Erwartet: [2, 5]