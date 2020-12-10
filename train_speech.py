import os
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio

import utils
from data import LyricsAlignDataset

from model_speech import SpeechRecognitionModel, data_processing

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                           100. * batch_idx * len(spectrograms) / data_len, loss.item()))


def main(learning_rate=5e-4, batch_size=20, epochs=10,
         experiment=Experiment(api_key='dummy_key', disabled=True)):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./checkpoint_lib"):
        os.makedirs("./checkpoint_lib")

    # train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
    # test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)
    #
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = data.DataLoader(dataset=train_dataset,
    #                                batch_size=hparams['batch_size'],
    #                                shuffle=True,
    #                                collate_fn=lambda x: data_processing(x, 'train'),
    #                                **kwargs)
    # test_loader = data.DataLoader(dataset=test_dataset,
    #                               batch_size=hparams['batch_size'],
    #                               shuffle=False,
    #                               collate_fn=lambda x: data_processing(x, 'valid'),
    #                               **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    train_data = LyricsAlignDataset({"train": [], "val": []}, "train", 22050, model.shapes, "hdf", dummy=False)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=hparams["batch_size"],
                                   collate_fn=lambda x: data_processing(x, 'train'),
                                   **kwargs)

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28, zero_infinity=True).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
        checkpoint_path = os.path.join('checkpoint_lib', "checkpoint_" + str(epoch))
        print("Saving model...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,  # state of training loop (was 'step')
        }, checkpoint_path)

        # test(model, device, test_loader, criterion, epoch, iter_meter, experiment)

if __name__ == '__main__':
    comet_api_key = ""  # add your api key here
    project_name = "speechrecognition"
    experiment_name = "speechrecognition-colab"

    if comet_api_key:
        experiment = Experiment(api_key=comet_api_key, project_name=project_name, parse_args=False)
        experiment.set_name(experiment_name)
        experiment.display()
    else:
        experiment = Experiment(api_key='dummy_key', disabled=True)

    learning_rate = 2e-4
    batch_size = 16
    epochs = 20

    main(learning_rate, batch_size, epochs, experiment)