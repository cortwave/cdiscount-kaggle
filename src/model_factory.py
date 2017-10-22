from torchvision.models import squeezenet1_1, resnet152, resnet34, resnet50, resnet101, densenet121, densenet161, densenet169, densenet201
from se_net.se_resnet import se_resnet18, se_resnet34, se_resnet50, se_resnet101, se_resnet152
import torch.nn as nn
from dpn import model_factory as dpn_factory
from se_net.se_inception import SEInception3

def get_model(num_classes, architecture):
    model = None
    if "seinception" in architecture:
        model = SEInception3(num_classes=num_classes)
    elif "seresnet" in architecture:
        if architecture == 'seresnet18':
            model = se_resnet18(num_classes).cuda()
        if architecture == 'seresnet34':
            model = se_resnet34(num_classes).cuda()
        if architecture == 'seresnet50':
            model = se_resnet50(num_classes).cuda()
        if architecture == 'seresnet101':
            model = se_resnet101(num_classes).cuda()
        if architecture == 'seresnet152':
            model = se_resnet152(num_classes).cuda()
    elif "resnet" in architecture:
        if architecture == 'resnet34':
            model = resnet34(pretrained=True).cuda()
        elif architecture == 'resnet50':
            model = resnet50(pretrained=True).cuda()
        elif architecture == 'resnet101':
            model = resnet101(pretrained=True).cuda()
        elif architecture == 'resnet152':
            model = resnet152(pretrained=True).cuda()
        if model is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
            model.avgpool = nn.AdaptiveAvgPool2d(1)
    elif "densenet" in architecture:
        if architecture == 'densenet121':
            model = densenet121(pretrained=True).cuda()
        elif architecture == "densenet161":
            model = densenet161(pretrained=True).cuda()
        elif architecture == "densenet169":
            model = densenet169(pretrained=True).cuda()
        elif architecture == "densenet201":
            model = densenet201(pretrained=True).cuda()
        if model is not None:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes).cuda()
    elif "squeezenet" in architecture:
        if architecture == "squeezenet1_1":
            model = squeezenet1_1(pretrained=True).cuda()
        if model is not None:
            final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
            model.num_classes = num_classes
    elif "dpn" in architecture:
        if architecture == "dpn68":
            model = dpn_factory.create_model(architecture,
                                               num_classes=1000,
                                               pretrained=True,
                                               test_time_pool=False)
            model.classifier = nn.Conv2d(model.in_chs, num_classes, kernel_size=1, bias=True)
    if model is None:
        raise Exception(f'Unknown architecture: {architecture}')
    return nn.DataParallel(model).cuda()