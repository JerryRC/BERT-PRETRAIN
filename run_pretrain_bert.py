import pickle
from tqdm import tqdm
import torch
import click
import os
from pytorch_transformers import BertConfig, BertForPreTraining
from create_pretraining_data import PreTrainingDataset


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tuning_one_epoch(model, optimizer, dataset, batch_size=16):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    total_loss = 0.0
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

        loss.backward()
        optimizer.step()


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-s", "--save", is_flag=True, help="Save the model files after every epoch")
@click.option("-d", "--data", help="Pretrain data's pickle file")
def main(save, data):

    f = open(data, 'rb')
    dataset = pickle.load(f)
    f.close()
    
    config = BertConfig.from_pretrained("bert-base-chinese")

    model = BertForPreTraining.from_pretrained("bert-base-chinese" ,config=config)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    tuning_one_epoch(model, optimizer, dataset, 8)


if __name__ == '__main__':
    main()
