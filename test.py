import torch
import yaml
import tqdm
from model.networklstm import create_model
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

print(f'{"="*30}{"BEGAIN":^20}{"="*30}')

dirname = 'test'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

config = yaml.load(open('./config-test.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device



class AmpData():
    def __init__(self, df, tokenizer_name='Rostlab/prot_bert_bfd', max_len=300):
       
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_len = max_len
        self.df = df
        
        self.seqs = self.get_seqs()
        
        
    def get_seqs(self):        
        # isolate the amino acid sequences and their respective AMP labels
        seqs = list(self.df['sequence']) 
        return seqs

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_len)
        
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}

        return sample
    
filepath = "./test.csv"
testdf = pd.read_csv(filepath, header=None)
testdf.columns = ['sequence']

test_dataset = AmpData(testdf)

sequences = []
attention_mask_test = []
test_inputs = []
for index, content in enumerate(test_dataset):
    attention_mask_test.append(content['attention_mask'])
    test_inputs.append(content['input_ids'])

attention_mask_test = np.array(attention_mask_test)
test_inputs = np.array(test_inputs)

sequences = test_dataset.get_seqs()
sequences = np.array(sequences)

class PeptideBERTDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

        self.length = len(self.input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        return {
            'input_ids': torch.tensor(input_id, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
    
test_dataset = PeptideBERTDataset(input_ids=test_inputs, attention_masks=attention_mask_test)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

print('Test dataset samples: ', len(test_dataset))
print('Test dataset batches: ', len(test_data_loader))
print()

model = create_model(config)
save_dir = config["model_path"]


def test(model, dataloader, device):
    model.eval()

    predictions = []
    predvalues = []
    allsequences = []

    for batch in tqdm(dataloader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.inference_mode():
            logits = model(inputs, attention_mask).squeeze()
    
        predvalues.extend(logits.cpu().tolist())
        preds = torch.where(logits > 0.5, 1, 0)
        predictions.extend(preds.cpu().tolist())
        allsequences.extend(batch['input_ids'].cpu().tolist())

    return predvalues, predictions, allsequences
################################################################################]
if not config['debug']:
    model.load_state_dict(torch.load(f'{save_dir}/model.pt')['model_state_dict'], strict=False)
allsequences, predvalues, predictions = test(model, test_data_loader, device)

resultdf = pd.DataFrame([allsequences, predvalues, predictions]).T
resultdf.columns = ['seq_ids', 'predvalue', 'predtype']

resultdf.to_csv(f'{save_dir}/{dirname}_pred_result.csv', index=None)
print('Model saved in: ', f'{save_dir}/{dirname}_pred_result.csv')
