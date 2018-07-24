

def init_trainer(opt, trainloader, validloader, model):

    model_name = opt.model

    if model_name.startswith('ssd'):
        from trainer.train_ssd import Trainer
        trainer = Trainer(opt,
                          train_dataloader=trainloader,
                          valid_dataloader=validloader,
                          model=model)

    elif model_name == 'fpnssd':
        from trainer.train_fpnssd import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    elif model_name == 'retinanet':
        from trainer.train_retinanet import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    elif model_name == 'refinedet':
        from trainer.train_refinedet import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    elif model_name == 's3fd':
        from trainer.train_s3fd import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    elif model_name == 'fpnssd_v_caption':
        from trainer.train_fpnssd_v_caption import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    elif model_name == 'resnet':
        from trainer.train_resnet import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    elif model_name == 'chanet':
        from trainer.train_chanet import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    elif model_name == 'resnet_nas':
        from trainer.train_resnet_nas import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    elif model_name == 'resnet_type':
        from trainer.train_resnet_type import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)
    elif model_name == 'resnet_num':
        from trainer.train_resnet_num import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)
    elif model_name == 'resnet_alp':
        from trainer.train_resnet_alp import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)
    elif model_name == 'resnet_alp2':
        from trainer.train_resnet_alp2 import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)
    elif model_name == 'resnet_sym':
        from trainer.train_resnet_sym import Trainer
        trainer = Trainer(opt,
                          trainloader,
                          validloader,
                          model)

    else:
        raise ValueError('not a valid model')

    return trainer