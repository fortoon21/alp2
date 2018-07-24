import torch
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn
from torch.nn.parameter import Parameter

def init_model(opt):

    model_name = opt.model


    if model_name == 'ssd300':
        from modellibs.ssd.model import SSD300
        assert opt.img_size == 300, 'image size does not match'

        model = SSD300(opt.num_classes)
        if opt.use_pretrained:
            state_dict = load_pretrained_weight_vgg('vgg16')
            #state_dict = torch.load('pretrained/vgg16-397923af.pth')
            model.extractor.load_state_dict(state_dict, strict=False)
            print("successfully load pretrained model from 'pretrained/vgg16-397923af.pth'")

    elif model_name == 'ssd512':
        from modellibs.ssd.model import SSD512
        assert opt.img_size == 512, 'image size does not match'
        model = SSD512(opt.num_classes)

    elif model_name == 'fpnssd':
        from modellibs.fpnssd.model import FPNSSD512
        assert opt.img_size == 512, 'image size must be 512'
        model = FPNSSD512(opt.num_classes)
        if opt.use_pretrained:
            state_dict = load_pretrained_weight_resnet('resnet50')
            model.load_state_dict(state_dict, strict=False)
            print("successfully load pretrained model from 'pretrained/resnet50-19c8e357'")

    elif model_name == 'retinanet':
        from modellibs.retinanet.model import RetinaNet
        assert opt.img_size == 512, 'image size must be 512'
        model = RetinaNet(opt.num_classes)
        if opt.use_pretrained:
            state_dict = load_pretrained_weight_resnet('resnet50')
            model.load_state_dict(state_dict, strict=False)
            print("successfully load pretrained model from 'pretrained/resnet50-19c8e357'")

    elif model_name == 'refinedet':
        from modellibs.refinedet.model import RefineDet
        assert opt.img_size == 320, 'image size must be 320'
        model = RefineDet(opt.img_size, opt.num_classes, True).to(opt.device)

        if opt.use_pretrained:
            pass

    elif model_name == 's3fd':
        from modellibs.s3fd.model import s3fd
        assert opt.img_size == 640, 'image size must be 640'
        model = s3fd()

        if opt.use_pretrained:
            state_dict = load_pretrained_weight_vgg('vgg16')

            local_state_dict = iter(model.state_dict().items())
            for i, (name, param) in enumerate(state_dict.items()):

                local_name, local_param = next(local_state_dict)
                try:
                    local_param.copy_(param)
                except:
                    pass
            print("successfully load pretrained model from 'pretrained/vgg16-397923af.pth'")

    elif model_name == 'fpnssd_v_caption':
        from modellibs.fpnssd_v_caption.model import FPNSSD_V_CAPTION
        model = FPNSSD_V_CAPTION(opt.num_classes)
        if opt.use_pretrained:
            state_dict = torch.load('pretrained/fpnssd512_20_trained.pth')
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name in own_state:
                    if name.split('.')[0] == 'cls_layers':
                        continue
                    if isinstance(param, Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    try:
                        if name.split('.')[0] == 'loc_layers':
                            name = name.split('.')[0] + '.' + str(int(name.split('.')[1]) + 1) + '.' + name.split('.')[2]
                        own_state[name].copy_(param)

                        if name == 'extractor.latlayer2.weight':
                            own_state['extractor.latlayer3.weight'].copy_(param[:, :256, :, :])
                        elif name == 'extractor.latlayer2.bias':
                            own_state['extractor.latlayer3.bias'].copy_(param)
                        elif name == 'loc_layers.1.weight':
                            own_state['loc_layers.0.weight'].copy_(param)
                        elif name == 'loc_layers.1.bias':
                            own_state['loc_layers.0.bias'].copy_(param)

                    except Exception:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))

            print("successfully load pretrained model from 'pretrained/fpnssd512_20_trained.pth'")

    elif model_name == 'resnet':
        if opt.resnet_model == 'resnet18':
            from modellibs.resnet.resnet import resnet18
            model = resnet18(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.resnet_model == 'resnet50':
            from modellibs.resnet.resnet import resnet50
            model = resnet50(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    elif model_name == 'chanet':
        if opt.base_model == 'resnet18':
            from modellibs.chanet.chanet import chanet18
            model = chanet18(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.base_model == 'resnet50':
            from modellibs.chanet.chanet import resnet50
            model = resnet50(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    elif model_name == 'resnet_nas':
        if opt.resnet_model == 'resnet18':
            from modellibs.resnet_nas.resnet_nas import resnet18_nas
            model = resnet18_nas(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.resnet_model == 'resnet50':
            from modellibs.resnet_nas.resnet_nas import resnet50_nas
            model = resnet50_nas(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    elif model_name == 'resnet_type':
        if opt.resnet_model == 'resnet18':
            from modellibs.resnet_type.resnet_type import resnet18_type
            model = resnet18_type(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.resnet_model == 'resnet50':
            from modellibs.resnet_type.resnet_type import resnet50_type
            model = resnet50_type(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    elif model_name == 'resnet_num':
        if opt.resnet_model == 'resnet18':
            from modellibs.resnet_num.resnet_num import resnet18_num
            model = resnet18_num(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.resnet_model == 'resnet50':
            from modellibs.resnet_num.resnet_num import resnet50_num
            model = resnet50_num(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    elif model_name == 'resnet_alp':
        if opt.resnet_model == 'resnet18':
            from modellibs.resnet_alp.resnet_alp import resnet18_alp
            model = resnet18_alp(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.resnet_model == 'resnet50':
            from modellibs.resnet_alp.resnet_alp import resnet50_alp
            model = resnet50_alp(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    elif model_name == 'resnet_alp2':
        if opt.resnet_model == 'resnet18':
            from modellibs.resnet_alp2.resnet_alp2 import resnet18_alp2
            model = resnet18_alp2(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.resnet_model == 'resnet50':
            from modellibs.resnet_alp2.resnet_alp2 import resnet50_alp2
            model = resnet50_alp2(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    elif model_name == 'resnet_sym':
        if opt.resnet_model == 'resnet18':
            from modellibs.resnet_sym.resnet_sym import resnet18_sym
            model = resnet18_sym(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.resnet_model == 'resnet50':
            from modellibs.resnet_sym.resnet_sym import resnet50_sym
            model = resnet50_sym(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    else:
        raise ValueError('Not implemented yet')

    if opt.device == 'cuda':
        model = model.to(opt.device)
        model = torch.nn.DataParallel(model, device_ids=opt.num_gpus)
        torch.backends.cudnn.benchmark = True

    return model


def load_pretrained_weight_vgg(pretrained_model):

    model_urls = {
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

    if pretrained_model.startswith('vgg'):
        state_dict = model_zoo.load_url(url=model_urls[pretrained_model], model_dir='pretrained')

    return state_dict


def load_pretrained_weight_resnet(pretrained_model):

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    if pretrained_model.startswith('resnet'):
        state_dict = model_zoo.load_url(url=model_urls[pretrained_model], model_dir='pretrained')

    return state_dict