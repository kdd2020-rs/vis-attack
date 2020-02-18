import random, os, math

from pprint import pprint

from experiments import Experimentation
from dataset import RecSysDataset
from models import VBPR, DeepStyle

class Args:
    def __init__(self, args):
        self.__dict__.update(args)

def load_model(dataset_name, algo_name, dataset):
  if algo_name == "vbpr":
    model = VBPR(dataset.n_users, dataset.n_items, dataset.corpus.image_features)
  elif algo_name == "deepstyle":
    model = DeepStyle(
      dataset.n_users, dataset.n_items, dataset.n_categories, 
      dataset.corpus.image_features, dataset.corpus.item_category
    )

  model.load(f'../data/dataset/{dataset_name}/models/{algo_name}_resnet50.pth')

  return model

def run_grid_search():
  from exps import gen_conf, grid_config_v1, base_config_v1, alreay_perfromed

  dataset_name = 'Electronics'
  algo_name = 'vbpr'

  experiments = Experimentation(load_model(dataset_name, algo_name), dataset_name)

  for name, args in gen_conf(grid_config_v1):
    if name not in alreay_perfromed:
      print(name)
      args.update(base_config_v1)
      pprint(args)
      experiments.run(name, Args(args))

  # with open("exp/exps.jsonl", "r") as f:
  #   for line in f:
  #     line = line.strip()
  #     if line.startswith("#"):
  #       continue
  #     args = Args(eval(line))
  #     exp.run(args)

default_args = {
  'k': 10,
  'k2': 10,
  'algorithm': 'vbpr',
  'do_pca': 0,
  'n_components': 128,
  'rank_distribution': 'uniform',
}

def run_from_rank_exp():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'single_user',
    'dataset_name': dataset_name,
    'steps': 20,
    'epsilon': 1/255,
    'gamma': 7,
    'by_rank': 1,
    'examples': 32,
  }

  args.update(default_args)

  exp_folder = f"exp_from_rank_range"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  random_users = random.sample(range(exp.model.n_users), k=50)
  rank_ranges = [1, 1000, 10000, 100000, 455412]
  for min_rank, max_rank in zip(rank_ranges[:-1], rank_ranges[1:]):
    from_rank = random.randint(min_rank, max_rank)
    args['from_rank'] = from_rank
    for blackbox in [0, 1]:
      args['blackbox'] = blackbox
      for user in random_users:
        args['user'] = user

        name = f"from_rank[{from_rank}, {blackbox}, {user}]"
        exp.run(f"{exp_folder}/{name}", Args(args))
        # im.save(f"exp/{name}.jpeg", "JPEG")

def run_main_table_exp():

  dataset_name = 'Electronics'
  algo_name = 'deepstyle'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'single_user',
    'dataset_name': dataset_name,
    'steps': 20,
    'epsilon': 1/255,
    'gamma': 7,
    'examples': 32,
  }

  args.update(default_args)

  exp_folder = f"exp_{dataset_name}_{algo_name}"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  repeat = 100

  random_users = random.sample(range(exp.model.n_users), k=repeat)
  random_items = random.sample(range(exp.model.n_items), k=repeat)
  for u, i in zip(random_users, random_items):
    args['user'] = u
    args['item'] = i
    for blackbox in [0, 1]:
      args['blackbox'] = blackbox
      if blackbox:
        for by_rank in [0, 1]:
          args['by_rank'] = by_rank
          name = f"main[{blackbox}:{by_rank}, {u}, {i}]"
          im = exp.run(f"{exp_folder}/{name}", Args(args))
          im.save(f"{exp_folder}/{name}.jpeg", "JPEG")
      else:
        name = f"main[{blackbox}, {u}, {i}]"
        im = exp.run(f"{exp_folder}/{name}", Args(args))
        im.save(f"{exp_folder}/{name}.jpeg", "JPEG")


def run_bb_examples_per_step_exp():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'single_user',
    'dataset_name': dataset_name,
    'epsilon': 1/255,
    'gamma': 7,
    'blackbox': 1,
    'by_rank': 1,
    'steps': 20,
  }

  args.update(default_args)

  exp_folder = f"exp_ex2step"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  repeat = 100
  budget = 4096

  # random_users = random.sample(range(exp.model.n_users), k=repeat)
  # random_items = random.sample(range(exp.model.n_items), k=repeat)
  # for u, i in zip(random_users, random_items):
  #   args['user'] = u
  #   args['item'] = i
  #   for p in range(1, int(math.log2(budget))):
  #     steps = 2**p
  #     examples = budget // steps
  #     args['examples'] = examples
  #     args['steps'] = steps
  #     name = f"main[{examples}, {steps}, {u}, {i}]"
  #     im = exp.run(f"{exp_folder}/{name}", Args(args))
  #     im.save(f"{exp_folder}/{name}.jpeg", "JPEG")

  random_users = random.sample(range(exp.model.n_users), k=repeat)
  random_items = random.sample(range(exp.model.n_items), k=repeat)
  for u, i in zip(random_users, random_items):
    args['user'] = u
    args['item'] = i
    for examples in [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]:
      steps = args['steps']
      args['examples'] = examples
      name = f"main[{examples}, {steps}, {u}, {i}]"
      im = exp.run(f"{exp_folder}/{name}", Args(args))
      # im.save(f"{exp_folder}/{name}.jpeg", "JPEG")


def run_segment_exp():

  dataset_name = 'Clothing_Shoes_and_Jewelry'
  algo_name = 'vbpr'

  dataset = RecSysDataset(dataset_name)
  model = load_model(dataset_name, algo_name, dataset)
  exp = Experimentation(model, dataset_name)

  seed = 0

  random.seed(seed)

  args = {
    'seed': seed,
    'experiment': 'segment',
    'dataset_name': dataset_name,
    'epsilon': 1/255,
    'gamma': 7,
    'by_rank': 1,
    "examples": 64,
    "steps": 20,
  }

  args.update(default_args)

  exp_folder = f"exp_segment_{dataset_name}_{algo_name}"
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

  random_items = random.sample(range(exp.model.n_items), k=100)
  for i in random_items:
    args['item'] = i
    for blackbox in [0]:
      args['blackbox'] = blackbox
      name = f"[{i, blackbox}]"
      exp.run(f"{exp_folder}/{name}", Args(args))
      exp.model = load_model(dataset_name, algo_name, dataset)


def main():
  # run_main_table_exp()
  # run_bb_examples_per_step_exp()
  run_segment_exp()
  # run_from_rank_exp()

if __name__ == '__main__':
  main()