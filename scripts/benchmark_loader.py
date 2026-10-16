import sys, time, json, logging
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from daisy.utils.loader import RawDataReader,Preprocessor
from daisy.utils.utils import get_ur
from daisy.utils.sampler import BasicNegtiveSampler
from daisy.utils.dataset import BasicDataset,get_dataloader
from daisy.model.SimpleXRecommender import SimpleX
config=json.loads(Path('artifacts/verified/simplex_yelp-200k/trial_configs/000.json').read_text())
config['logger']=logging.getLogger('benchmark')
processor=Preprocessor(config)
df=processor.process(RawDataReader(config).get_data()).iloc[:63201].copy()
config['user_num'],config['item_num']=processor.user_num,processor.item_num
config['train_ur']=get_ur(df)
samples=BasicNegtiveSampler(df,config).sampling()
for threads in [1,2]:
 torch.set_num_threads(threads)
 for vector in [False,True]:
  model=SimpleX(config)
  optimizer=model._build_optimizer()
  loader=get_dataloader(BasicDataset(samples),config['batch_size'],True,0)
  iterator=iter(loader)
  rows=torch.as_tensor(samples)
  started=time.perf_counter()
  for i in range(100):
   batch=rows[i*config['batch_size']:(i+1)*config['batch_size']].T if vector else next(iterator)
   optimizer.zero_grad()
   loss=model.calc_loss(batch)
   loss.backward()
   optimizer.step()
  print({'threads':threads,'vector_batch':vector,'seconds_per_100_steps':time.perf_counter()-started},flush=True)
