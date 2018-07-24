import os
import time
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from loss.ssd_loss import SSDLoss
from metrics.voc_eval import voc_eval
from modellibs.s3fd.box_coder import S3FDBoxCoder
from utils.average_meter import AverageMeter


class Trainer(object):

    def __init__(self, opt, train_dataloader, valid_dataloader, model):
        self.opt = opt
        self.current_lr = opt.lr
        self.start_epoch = opt.start_epochs

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.max_iter_train = len(self.train_dataloader)
        self.max_iter_valid = len(self.valid_dataloader)

        self.model = model

        self.criterion = torch.nn.CrossEntropyLoss().cuda()

        self.optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr, weight_decay=opt.weight_decay)

        self.best_loss = float('inf')

        if opt.resume:
            self.optimizer.load_state_dict(torch.load(opt.resume_path)['optimizer'])

    def train_model(self, max_epoch, learning_rate, layers=None):

        self.max_epoch = max_epoch

        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch in self.opt.lr_steps:
                self.opt.step_index += 1
                self.adjust_learning_rate(self.opt.gamma, self.opt.step_index)

            self.train_epoch(epoch)
            self.valid_epoch(epoch)

        print('')
        print('optimization done')
        save_dir = 'experiments/%s_%s_%s' % (self.opt.dataset, self.opt.task, self.opt.model)
        file_name = '%s_%s_%s_best_loss_%f' % (self.opt.dataset, self.opt.task, self.opt.model, self.best_loss)
        os.rename(self.opt.expr_dir,  os.path.join(save_dir,file_name))

    def train_epoch(self, epoch):
        """ training """
        self.model.train()
        self.optimizer.zero_grad()

        train_loss = 0
        for batch_idx, (inputs, label) in enumerate(self.train_dataloader):
            inputs = inputs.to(self.opt.device)
            label = label.to(self.opt.device)

            output = self.model(inputs)
            loss = self.criterion(output, label)

            loss.backward()
            if ((batch_idx + 1) % self.opt.accum_grad == 0) or ((batch_idx+1) == self.max_iter_train):
                self.optimizer.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()

            train_loss += loss.item()
            if batch_idx % self.opt.print_freq == 0:
                print('Epoch[%d/%d] Iter[%d/%d] Learning Rate: %.6f Loss[Avg]: %.4f[%.4f]' %
                      (epoch, self.max_epoch, batch_idx, self.max_iter_train, self.current_lr, loss.item(), train_loss/(batch_idx+1)))

    def valid_epoch(self, epoch):

        batch_time = AverageMeter()

        total_losses = AverageMeter()

        top1 = AverageMeter()
        top5 = AverageMeter()

        correct = 0

        """ validate """
        self.model.eval()
        test_loss = 0

        end = time.time()
        for batch_idx, (inputs, label) in enumerate(self.valid_dataloader):
            with torch.no_grad():
                inputs = inputs.to(self.opt.device)
                label = label.to(self.opt.device)

                output = self.model(inputs)
                loss = self.criterion(output, label)

                total_losses.update(loss.item(), inputs.size(0))

                pred = output.data.max(1, keepdim=True)[1].cpu()
                correct += pred.eq(label.cpu().view_as(pred)).sum()
                batch_time.update(time.time() - end)
                end = time.time()

                test_loss += loss.item()

                if batch_idx % self.opt.print_freq_eval == 0:
                    print('Validation[%d/%d] Total Loss: %.4f[%.4f]' %
                          (batch_idx, len(self.valid_dataloader), loss.item(), test_loss/(batch_idx+1)))

        num_test_data = len(self.valid_dataloader.dataset)
        accuracy = 100. * correct / num_test_data

        test_loss /= len(self.valid_dataloader)
        if test_loss < self.best_loss:
            print('Saving..')

            state = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': test_loss,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(self.opt.expr_dir, 'model_best.pth'))
            self.best_loss = test_loss
        print('[*] Model %s,\tCurrent Loss: %f\tBest Loss: %f' % (self.opt.model, test_loss, self.best_loss))
        print('Val Accuracy: {}/{} ({:.0f}%)'.format(
            correct, num_test_data, accuracy))

    def adjust_learning_rate(self, gamma, step):

        self.current_lr = self.opt.lr * (gamma ** (step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def make_dir(self, dir_path):
        if not os.path.exists(os.path.join(self.opt.expr_dir, dir_path)):
            os.mkdir(os.path.join(self.opt.expr_dir, dir_path))



