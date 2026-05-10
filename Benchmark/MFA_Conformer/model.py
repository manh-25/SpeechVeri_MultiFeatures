import torch
import torch.nn as nn
import os
import platform
from contextlib import contextmanager


@contextmanager
def _windows_safe_wenet_import():
    if platform.system().lower() != "windows":
        yield
        return

    original_popen = os.popen

    def _patched_popen(cmd, *args, **kwargs):
        if isinstance(cmd, str) and "lscpu" in cmd:
            # Wenet probes Linux CPU vendor during import; mute this probe on Windows.
            return original_popen("echo", *args, **kwargs)
        return original_popen(cmd, *args, **kwargs)

    os.popen = _patched_popen
    try:
        yield
    finally:
        os.popen = original_popen


with _windows_safe_wenet_import():
    from wenet.models.transformer.encoder import ConformerEncoder


from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d


class _FallbackAttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, bottleneck_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Conv1d(channels, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, channels, kernel_size=1)

    def forward(self, x):
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        out = torch.cat([mean, std], dim=1)
        return out.unsqueeze(-1)


class _FallbackConformerEncoder(nn.Module):
    def __init__(self, input_size: int, num_blocks: int, output_size: int):
        super().__init__()
        self.in_proj = nn.Linear(input_size, output_size)
        layer = nn.TransformerEncoderLayer(
            d_model=output_size,
            nhead=4,
            dim_feedforward=output_size * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)

    def forward(self, feat, lens):
        x = self.in_proj(feat)
        x = self.encoder(x)
        return x, None

class Conformer(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos"):
        super(Conformer, self).__init__()
        self.conformer = ConformerEncoder(
            input_size=n_mels,
            num_blocks=num_blocks,
            output_size=output_size,
            input_layer=input_layer,
            pos_enc_layer_type=pos_enc_layer_type,
        )

        self.pooling = AttentiveStatisticsPooling(output_size)
        self.bn = BatchNorm1d(input_size=output_size * 2)

        self.fc = torch.nn.Linear(output_size*2, embedding_dim)
    
    def forward(self, feat):
        feat = feat.squeeze(1).permute(0, 2, 1)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens*feat.shape[1]).int()
        x, masks = self.conformer(feat, lens)
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

def conformer(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=192, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
    model = Conformer(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size, 
            embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model




if __name__ == "__main__":
    for i in range(6, 7):
        print("num_blocks is {}".format(i))
        model = conformer(num_blocks=i)

        import time
        model = model.eval()
        time1 = time.time()
        with torch.no_grad():
            for i in range(100):
                data = torch.randn(1, 1, 80, 500)
                embedding = model(data) 
        time2 = time.time()
        val = (time2 - time1)/100
        rtf = val / 5

        total = sum([param.nelement() for param in model.parameters()])
        print("total param: {:.2f}M".format(total/1e6))
        print("RTF {:.4f}".format(rtf))