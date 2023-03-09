import os
import pickle
from time import time

import click
import torch
import torch.utils.data
from loguru import logger
from pytorch_transformers import BertConfig, BertForPreTraining
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from create_pretraining_data import PreTrainingDataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tuning_one_epoch(save, model, optimizer, dataset, writer, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    total_loss = 0.0
    batch_n = 0
    for input_ids, segment_ids, input_mask, masked_lm_ids, next_sentence_label in tqdm(generator):
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_mask = input_mask.to(device)
        masked_lm_ids = masked_lm_ids.to(device)
        next_sentence_label = next_sentence_label.to(device)

        optimizer.zero_grad()
        loss, _, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                           masked_lm_labels=masked_lm_ids, next_sentence_label=next_sentence_label)

        total_loss += loss.item()

        writer.add_scalar('Loss/train/shui5', loss.item(), batch_n)

        loss.backward()
        optimizer.step()

        batch_n += 1

    total_loss /= batch_n

    if save:
        torch.save(model, 'finetune.pt')

    return total_loss


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-s", "--save", is_flag=True, help="Save the model files after every epoch")
@click.option("-d", "--data", help="Pretrain data's pickle file")
def main(save, data):
    if not os.path.exists(data):
        logger.error("pickle file does not exist.")
        return

    s_t = time()
    logger.info('Preparing dataset: %s' % data)
    with open(data, 'rb') as f:
        dataset = pickle.load(f)
        assert isinstance(dataset, PreTrainingDataset)
    logger.info('Dataset loading completed, taking {} (s)'.format(time() - s_t))

    logger.info('Preparing model')
    s_t = time()
    config = BertConfig.from_pretrained("bert-base-chinese")

    model = BertForPreTraining.from_pretrained("bert-base-chinese", config=config)
    # model = torch.load('finetune.pt')
    model = model.to(device)
    logger.info('Model loaded, taking {} (s)'.format(time() - s_t))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    logger.info('Start training')
    with SummaryWriter('logs') as writer:
        loss = tuning_one_epoch(save, model, optimizer, dataset, writer, 32)
    logger.info('Final loss is %f' % loss)


if __name__ == '__main__':
    main()
