# coding: UTF-8
import time
import torch
import numpy as np
# from train_eval import train, init_network
from importlib import import_module
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
import torch.nn.functional as F


import pandas as pd
import numpy as np
import sklearn
# text preprocessing
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import torch.utils.data as Data

import os
import random
from importlib import import_module


from utils import load_data, make_data
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser(description='English Text Classification')
# parser.add_argument('--model', type=str, default='CNN_LSTM', help='choose a model: CNN_LSTM, TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att')
parser.add_argument('--embedding', type=int, default='100', help='random or pre_trained')
parser.add_argument('--word', type=bool, default=False, help='True for word, False for char')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--require_improvement', type=int, default=10, help='rquirement')
parser.add_argument('--num_epochs', type=int, default=50, help='epochs')
parser.add_argument('--pad_size', type=int, default=32, help='pad size')

parser.add_argument('--train_path', type=str, default='../data_final/data_train.csv', help='train_path')
parser.add_argument('--test_path', type=str, default='../data_final/data_test.csv', help='test_path')
parser.add_argument('--vocab_path', type=str, default="../data_final/vocab.pkl", help='vocab_path')


args = parser.parse_args()




def test(config, model, test_iter):
    model.eval()
    # loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    # predict_all_mul = np.array([], dtype=int)
    # labels_all_mul = np.array([], dtype=int)
    predict_all_mul = []
    labels_all_mul = []

    with torch.no_grad():
        for texts, labels, labels_mul in test_iter:
            outputs, outputs_2, _ = model(texts)
            # loss = F.cross_entropy(outputs, labels)
            # loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            labels_mul = labels_mul.data.cpu().numpy().astype(int)
            predic_mul = outputs_2.cpu().numpy()
            predic_mul = np.around(predic_mul,0).astype(int)

            # labels_all_mul = np.append(labels_all_mul, labels_mul)
            # predict_all_mul = np.append(predict_all_mul, predic_mul)
            labels_all_mul.append(labels_mul)
            predict_all_mul.append(predic_mul)

            # pre_mul = precision_score(labels_all, predict_all)
            # rec_mul = recall_score(labels_all, predict_all)
            # f1_mul = f1_score(labels_all, predict_all)
            # HL_mul = hamming_loss(labels_all, predict_all)
            # print("labels_single", pre_mul, rec_mul, labels_all_mul, predict_all_mul)






    # acc = accuracy_score(labels_all, predict_all)
    pre = precision_score(labels_all, predict_all, average='micro')
    rec = recall_score(labels_all, predict_all, average='micro')
    f1 = f1_score(labels_all, predict_all, average='micro')
    HL = hamming_loss(labels_all, predict_all)


    # print("labels", labels_all_mul, predict_all_mul)
    # labels_all_mul = np.array(labels_all_mul, dtype=np.int)
    # predict_all_mul = np.array(predict_all_mul, dtype=np.int)
    pre_mul = precision_score(labels_all_mul[0], predict_all_mul[0], average='samples')
    rec_mul = recall_score(labels_all_mul[0], predict_all_mul[0], average='samples')
    f1_mul = f1_score(labels_all_mul[0], predict_all_mul[0], average='samples')
    HL_mul = hamming_loss(labels_all_mul[0], predict_all_mul[0])


    # if test:
    #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion
    return pre, rec, f1, HL, pre_mul, rec_mul, f1_mul, HL_mul






from gensim.models import word2vec


BCE_lo = torch.nn.BCELoss()
if __name__ == '__main__':
    # Set random number seed
    setup_seed(110)  ## 110\112\115
    model_pre = word2vec.Word2Vec.load("../pre_crowdTasks")
    maxlen = 100
    # print("dd", model['Collect'])


    ### From original content to participle content
    print("loading data......")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    train_data = load_data(train_df, model_pre)
    test_data = load_data(test_df, model_pre)

    train_inputs, train_labels, train_labels_mul, \
    test_inputs, test_labels, test_lables_mul, vocab_size = make_data(train_data, test_data)

    train_inputs, train_labels = torch.LongTensor(train_inputs), torch.LongTensor(train_labels)
    train_labels_mul, test_lables_mul = torch.FloatTensor(train_labels_mul), torch.FloatTensor(test_lables_mul)
    test_inputs, test_labels = torch.LongTensor(test_inputs), torch.LongTensor(test_labels)

    # Load the training dataset
    train_dataset = Data.TensorDataset(train_inputs, train_labels, train_labels_mul)
    train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Load test dataset
    test_dataset = Data.TensorDataset(test_inputs, test_labels, test_lables_mul)
    test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=True)

     # =   # Number of dictionaries


    ### load model
    print("loading model......")
    # from models.TextCNN import Model
    # from models.TextRNN import Model
    # from models.TextRNN_Att import Model
    from models.BiMCNN import Model

    model = Model(n_vocab=vocab_size, embed=100, num_classes=4, num_class_2=14)
    ## model optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ### start training
    total_batch = 0  # How many batches are recorded
    best_f1 = 0.0
    last_improved = 0  # Record the batch number of the last verification set loss drop
    flag = False  # Whether the record has not improved for a long time
    Loss_list = []

    # writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(args.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        # scheduler.step() # learning rate decay
        for i, (trains, labels, labels_mul) in enumerate(train_loader):
            # print("trains", trains)
            outputs, outputs_2, alpha = model(trains)
            model.zero_grad()
            # print("label", outputs, labels)

            loss_1 = F.cross_entropy(outputs, labels)
            # print("output", outputs_2, labels_mul)
            loss_2 = BCE_lo(outputs_2, labels_mul)

            w2 = torch.cosine_similarity(outputs_2, labels_mul)
            # w1 = torch.cosine_similarity(outputs, labels)
            # print("w1", w1, w1.shape)
            # print("w2", w2, w2.shape, loss_2.shape)
            # loss = loss_1/(2.0+torch.mean(w2)) + loss_2*(1.0+torch.mean(w2))/(2.0+torch.mean(w2))
            # loss = 0.2 * loss_1 + 0.8 * loss_2
            # loss = 0.4 * loss_1 + 0.6 * loss_2
            # loss = 0.5 * loss_1 + 0.5 * loss_2
            loss = 0.7 * loss_1 + 0.3 * loss_2
            loss.backward()
            optimizer.step()

            Loss_list.append(loss)


        ### start testing
        pre, rec, f1, HL, pre_mul, rec_mul, f1_mul, HL_mul = test(args, model, test_loader)
        f11 = open('results.txt', 'a+')
        f11.write(str(epoch) + '\t' + str(pre) + '\t' + str(rec) + '\t' + str(f1) + '\t' +
                  str(HL) + '\t' + str(pre_mul) + '\t' + str(rec_mul) + '\t' + str(f1_mul) + '\t' +
                  str(HL_mul) + '\n')

        # If the validation accuracy improves, replace it with the best result and save the model
        if f1_mul >= best_f1:
            best_f1 = f1_mul
            last_improved = epoch
            improved_str = 'improved!'
            print('saving model_att...')
            # path = 'model/{}_Our_VAE_000'.format(dataset_name)
            # cVAE_model.cpu()
            # torch.save(cVAE_model.state_dict(), path)
            np.save('results_shuzu_73.npy', Loss_list)
            np.save('alpha.npy', alpha.detach().numpy())

        else:
            improved_str = ''


        # If there is no improvement after 1000 steps, stop training.
        print("last_improved", epoch, last_improved, args.require_improvement)
        if epoch - last_improved > args.require_improvement:
            print("No optimization for ", args.require_improvement, " steps, auto-stop in the ", epoch, " step!")
            # break out of this cycle
            flag = True
            break
        # Break out of the loop for all training epochs
        if flag:
            break

