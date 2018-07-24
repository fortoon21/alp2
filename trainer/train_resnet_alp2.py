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

        self.criterion_alp = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_sb = torch.nn.CrossEntropyLoss().cuda()

        self.optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

        self.best_loss = float('inf')

        if opt.resume:
            self.optimizer.load_state_dict(torch.load(opt.resume_path)['optimizer'])

    def train_model(self, max_epoch, learning_rate, layers=None):

        self.max_epoch = max_epoch

        for epoch in range(self.start_epoch, self.max_epoch):
            self.adjust_learning_rate(self.optimizer, epoch)

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
        for batch_idx, (inputs, labels) in enumerate(self.train_dataloader):
            label_alp = labels[0]
            label_sb = labels[1]


            inputs = inputs.to(self.opt.device)
            label_alp = label_alp.to(self.opt.device)
            label_sb = label_sb.to(self.opt.device)


            output_alp, output_sb = self.model(inputs)
            loss_alp = self.criterion_alp(output_alp, label_alp)
            loss_sb = self.criterion_sb(output_sb, label_sb)
            loss = loss_alp + loss_sb

            loss.backward()
            if ((batch_idx + 1) % self.opt.accum_grad == 0) or ((batch_idx+1) == self.max_iter_train):
                self.optimizer.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()

            train_loss += loss.item()
            if batch_idx % self.opt.print_freq == 0:
                print('Epoch[%d/%d] Iter[%d/%d] Learning Rate: %.6f Total Loss: %.4f, Alphabet Loss: %.4f, SmallBig Loss: %.4f' %
                      (epoch, self.max_epoch, batch_idx, self.max_iter_train, self.current_lr, loss.item(), loss_alp.item(), loss_sb.item()))

    def valid_epoch(self, epoch):

        correct_alp = 0
        correct_sb = 0

        """ validate """
        self.model.eval()
        test_loss = 0

        for batch_idx, (inputs, labels) in enumerate(self.valid_dataloader):
            with torch.no_grad():
                label_alp = labels[0]
                label_sb = labels[1]

                inputs = inputs.to(self.opt.device)
                label_alp = label_alp.to(self.opt.device)
                label_sb = label_sb.to(self.opt.device)

                output_alp, output_sb = self.model(inputs)
                loss_alp = self.criterion_alp(output_alp, label_alp)
                loss_sb = self.criterion_sb(output_sb, label_sb)
                loss = loss_alp + loss_sb

                pred_alp = output_alp.data.max(1, keepdim=True)[1].cpu()
                pred_sb = output_sb.data.max(1, keepdim=True)[1].cpu()
                correct_alp += pred_alp.eq(label_alp.cpu().view_as(pred_alp)).sum()
                correct_sb += pred_sb.eq(label_sb.cpu().view_as(pred_sb)).sum()

                test_loss += loss.item()

                if batch_idx % self.opt.print_freq_eval == 0:
                    print('Validation[%d/%d] Total Loss: %.4f, Alphabet Loss: %.4f, SmallBig Loss: %.4f' %
                          (batch_idx, len(self.valid_dataloader), loss.item(), loss_alp.item(), loss_sb.item()))

        num_test_data = len(self.valid_dataloader.dataset)
        accuracy_alp = 100. * correct_alp / num_test_data
        accuracy_sb = 100. * correct_sb / num_test_data

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
        print('Val Accuracy_Alphabet: {}/{} ({:.0f}%) | Val Accuracy_SmallBig: {}/{} ({:.0f}%)\n'.format(
            correct_alp, num_test_data, accuracy_alp,
            correct_sb, num_test_data, accuracy_sb))

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.current_lr = self.opt.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def make_dir(self, dir_path):
        if not os.path.exists(os.path.join(self.opt.expr_dir, dir_path)):
            os.mkdir(os.path.join(self.opt.expr_dir, dir_path))
