import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model(args, backbone_name="resnet18_cub", full_model=False):
    if "clip" in backbone_name:
        import clip
        # We assume clip models are passed of the form : clip:RN50
        clip_backbone_name = backbone_name.split(":")[1]
        backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=args.out_dir)
        backbone = backbone.eval()
        model = None
    
    elif backbone_name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=args.out_dir)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])
    
    elif backbone_name.lower() == "ham10000_inception":
        from .derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(args, backbone_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
        
    elif backbone_name == "resnet18":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        from .pretrain_audio import finetune_resnet
        import os.path
        model = ptcv_get_model(backbone_name, pretrained=True, root=args.out_dir)

        if not os.path.isfile(os.path.join(args.out_dir, "resnet18_ESC50.pth")):    # finetune
            model = finetune_resnet(model, args)
        else:
            model.load_state_dict(torch.load(os.path.join(args.out_dir, "resnet18_ESC50.pth")))    # load existing model
        backbone = ResNetBottom(model)
        preprocess = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif backbone_name == "resnet34":
        from .pretrain_audio import finetune_resnet
        import os.path
        from torchvision.models import resnet34
        
        # if args.dataset == 'esc-50':
        #     datapath = "esc50resnet34.pth"
        # else:
        #     datapath = "FSD_resnet34.pth"

        if not os.path.isfile(os.path.join(args.out_dir, "FSD_resnet34.pth")):    # finetune
            model = resnet34(pretrained=True)
            model.fc = nn.Linear(512,50)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model = model.to(args.device)
            model = finetune_resnet(model, args)
        else:
            model = torch.load(os.path.join(args.out_dir, "FSD_resnet34.pth"))    # load existing model
            model = model.to(args.device)
        backbone = ResNetBottom(model)
        preprocess = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif backbone_name == "HTSAT":
        from transformers import ClapAudioModel
        model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused").audio_encoder
        # model.load_state_dict(torch.load("C:/Users/lenna/Documents/UvA/FACT/post-hoc-cbm/data/HTSAT_ESC_exp=1_fold=0_acc=0.970.ckpt"))
        # model = torch.load(os.path.join(args.out_dir, "FSD_resnet34.pth"))    # load existing model
        model = model.to(args.device)
        backbone = ResNetBottom(model)
        preprocess = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess


