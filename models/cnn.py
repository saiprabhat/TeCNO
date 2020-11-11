import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
from datetime import datetime

### TWO HEAD MODELS ###


class TwoHeadResNet50Model(nn.Module):
    def __init__(self, hparams):
        """Constructor of the class. 

        Args:
            hparams (namespace): Dictionary containing several key-value pairs.
                                The keys `pretrained` and `out_features` are used here.
        """
        super(TwoHeadResNet50Model, self).__init__()
        self.model = models.resnet50(pretrained=hparams.pretrained)
        # Replace final layer of ResNet50 with number of labels
        self.model.fc = Identity()
        self.fc_phase = nn.Linear(2048, hparams.out_features)
        self.fc_tool = nn.Linear(2048, hparams.out_features)

    def forward(self, x):
        """Performs a forward pass through a TwoHeadResNet50 model and computes the predictions for each frame of a video.

        Args:
            x ([FloatTensor]): 

        Returns:
            [type]: [description]
        """
        now = datetime.now()
        out_stem = self.model(x)
        phase = self.fc_phase(out_stem)
        tool = self.fc_tool(out_stem)
        return out_stem, phase, tool

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser


#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
