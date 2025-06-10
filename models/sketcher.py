import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
# from models.pidinet import models
# from models.pidinet.models.convert_pidinet import convert_pidinet



class Sketcher(nn.Module):
    def __init__(self, device):
        super(Sketcher, self).__init__()
        sa = True
        dil = True
        config = "carv4"
        self.model = models.pidinet_converted(config, dil, sa).to(device)
        self.model.eval()
        checkpoint = torch.load(os.path.join(PROJECT_ROOT, 'data', 'pidinet_checkpoints', 'table5_pidinet.pth'),
                                map_location=device)

        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        self.model.load_state_dict(convert_pidinet(checkpoint, "carv4"))

    def forward(self, image):
        results = self.model(image)
        result = results[-1]

        return result
