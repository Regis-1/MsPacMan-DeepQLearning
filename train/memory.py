import numpy as np
import random
import h5py
import os
import pickle

from sum_tree import SumTree

#REPLAY MEMORY
class ReplayMemory:
    MAX_SIZE = 10000
    STATE_TYPE = 'uint8'


    def __init__(self,
                 input_height,
                 input_width,
                 input_channels,
                 max_size=MAX_SIZE,
                 state_type=STATE_TYPE):
        self.cur_idx = 0
        self.is_full = False
        self.max_size = max_size
        
        self.states = np.zeros((self.max_size,
                                input_height,
                                input_width,
                                input_channels),
                                dtype=state_type)
        self.actions = np.zeros((self.max_size,),
                                dtype='uint8')
        self.rewards = np.zeros((self.max_size,),
                                dtype='uint8')
        self.next_states = np.zeros((self.max_size,
                                     input_height,
                                     input_width,
                                     input_channels),
                                    dtype=state_type)
        self.continues = np.zeros((self.max_size,),
                                   dtype='bool')
        self.losses = np.zeros((self.max_size,),
                                dtype='float16')


    def clear(self):
        self.cur_idx = 0
        self.is_full = False


    def append(self, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        self.set(self.cur_idx,
                 state=state,
                 action=action,
                 reward=reward,
                 next_state=next_state,
                 cont=cont,
                 loss=loss)

        self.increment_idx()


    def extend(self, target):
        for i in range(len(target)):
            target.copy(i, self, self.cur_idx)
            self.increment_idx()


    def get(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
            'cont': self.continues[idx],
            'loss': self.losses[idx]
        }


    def set(self, idx, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        if state is not None:
            self.states[idx] = state
        if action is not None:
            self.actions[idx] = action
        if reward is not None:
            self.rewards[idx] = reward
        if next_state is not None:
            self.next_states[idx] = next_state
        if cont is not None:
            self.continues[idx] = cont
        if loss is not None:
            self.losses[idx] = loss


    def copy(self, idx, target, target_idx):
        target.states[target_idx] = self.states[idx]
        target.actions[target_idx] = self.actions[idx]
        target.rewards[target_idx] = self.rewards[idx]
        target.next_states[target_idx] = self.next_states[idx]
        target.continues[target_idx] = self.continues[idx]
        target.losses[target_idx] = self.losses[idx]


    def increment_idx(self):
        if self.cur_idx + 1 >= self.max_size:
            self.is_full = True
        self.cur_idx = (self.cur_idx + 1) % self.max_size


    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get(idx)
        else:
            if idx.start is None:
                start = 0
            else:
                start = idx.start

            if idx.stop is None:
                stop = self.max_size
            else:
                stop = idx.stop

            if idx.step is None:
                step = 1
            else:
                step = idx.step

            return [self[i] for i in range(start, stop, step)]


    def __len__(self):
        if self.is_full:
            return self.max_size

        return self.cur_idx


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
#REPLAY MEMORY DISK
class ReplayMemoryDisk:
    MAX_SIZE = 2000000
    CACHE_SIZE = 350000


    def __init__(self,
                 fn,
                 input_height=1,
                 input_width=1,
                 input_channels=1,
                 state_type='uint8',
                 max_size=MAX_SIZE,
                 cache_size=CACHE_SIZE):
        if not os.path.exists(fn):
            init = True
        else:
            init = False

        self.filename = fn
        self.data_file = h5py.File(fn, 'a')

        if init:
            self.create_dataset(input_height, input_width, input_channels, state_type, max_size)

            self._cur_idx = 0
            self._is_full = False
            self.max_size = max_size
        else:
            self._cur_idx = self.data_file.attrs['cur_idx']
            self._is_full = self.data_file.attrs['is_full']
            self.max_size = self.data_file.attrs['max_size']

            input_height = self.data_file.attrs['input_height']
            input_width = self.data_file.attrs['input_width']
            input_channels = self.data_file.attrs['input_channels']
            state_type = self.data_file.attrs['state_type']

        # init cache
        if cache_size:
            self.cache = ReplayMemory(input_height,
                                      input_width,
                                      input_channels,
                                      max_size=cache_size,
                                      state_type=state_type)

            self.cache_map = {}
            self.cache_map_rev = {}
        else:
            self.cache = None


    def create_dataset(self, input_height, input_width, input_channels, state_type, max_size):
        self.data_file.attrs['cur_idx'] = 0
        self.data_file.attrs['is_full'] = False
        self.data_file.attrs['max_size'] = max_size
        self.data_file.attrs['input_height'] = input_height
        self.data_file.attrs['input_width'] = input_width
        self.data_file.attrs['input_channels'] = input_channels
        self.data_file.attrs['state_type'] = state_type


        self.data_file.create_dataset('states',
                                      shape=(max_size, input_height, input_width, input_channels),
                                      dtype=state_type)
        self.data_file.create_dataset('actions',
                                      shape=(max_size,),
                                      dtype='uint8')
        self.data_file.create_dataset('rewards',
                                      shape=(max_size,),
                                      dtype='uint8')
        self.data_file.create_dataset('next_states',
                                      shape=(max_size, input_height, input_width, input_channels),
                                      dtype=state_type)
        self.data_file.create_dataset('continues',
                                      shape=(max_size,),
                                      dtype='bool')
        self.data_file.create_dataset('losses',
                                      shape=(max_size,),
                                      dtype='float16')

    def clear(self):
        self.cur_idx = 0
        self.is_full = False

        if self.cache:
            self.cache.clear()


    def load_memory(self, memory_fn, delete=True):
        with open(memory_fn, 'rb') as fin:
            size = pickle.load(fin)

            for i in range(size):
                self.append(*pickle.load(fin))

        if delete:
            os.unlink(memory_fn)


    def append(self, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        self.set(self.cur_idx,
                 state=state,
                 action=action,
                 reward=reward,
                 next_state=next_state,
                 cont=cont,
                 loss=loss)

        self.increment_idx()


    def extend(self, target):
        for i in range(len(target)):
            target.copy(i, self, self.cur_idx)
            self.increment_idx()


    def set(self, idx, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        if state is not None:
            self.data_file['states'][idx] = state
        if action is not None:
            self.data_file['actions'][idx] = action
        if reward is not None:
            self.data_file['rewards'][idx] = reward
        if next_state is not None:
            self.data_file['next_states'][idx] = next_state
        if cont is not None:
            self.data_file['continues'][idx] = cont
        if loss is not None:
            self.data_file['losses'][idx] = loss


        if self.cache:
            cache_idx = self.cache_map.get(idx, None)
            if cache_idx is not None:
                del self.cache_map[idx]
                del self.cache_map_rev[cache_idx]


    def get(self, idx):
        if self.cache:
            cache_idx = self.cache_map.get(idx, None)

            if cache_idx is not None:
                return self.cache.get(cache_idx)

        row = self.get_row(idx)

        if self.cache:
            self.cache_row(idx, **row)

        return row


    def copy(self, idx, target, target_idx):
        if self.cache:
            cache_idx = self.cache_map.get(idx, None)

            if cache_idx is not None:
                target.states[target_idx] = self.cache.states[cache_idx]
                target.actions[target_idx] = self.cache.actions[cache_idx]
                target.rewards[target_idx] = self.cache.rewards[cache_idx]
                target.next_states[target_idx] = self.cache.next_states[cache_idx]
                target.continues[target_idx] = self.cache.continues[cache_idx]
                target.losses[target_idx] = self.cache.losses[cache_idx]

                return


        target.states[target_idx] = self.states[idx]
        target.actions[target_idx] = self.actions[idx]
        target.rewards[target_idx] = self.rewards[idx]
        target.next_states[target_idx] = self.next_states[idx]
        target.continues[target_idx] = self.continues[idx]
        target.losses[target_idx] = self.losses[idx]


        if self.cache:
            row = self.get_row(idx)
            self.cache_row(idx, row)


    def close(self):
        self.data_file.close()


    def get_row(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
            'cont': self.continues[idx],
            'loss': self.losses[idx]
        }


    def cache_row(self, idx, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        cache_cur_idx = self.cache.cur_idx
        self.cache_map[idx] = cache_cur_idx
        self.cache.append(state=state,
                          action=action,
                          reward=reward,
                          next_state=next_state,
                          cont=cont,
                          loss=loss)

        old_idx = self.cache_map_rev.get(cache_cur_idx, None)
        if old_idx is not None:
            del self.cache_map[old_idx]

        self.cache_map_rev[cache_cur_idx] = idx


    def increment_idx(self):
        if self.cur_idx + 1 >= self.max_size:
            self.is_full = True
        self.cur_idx = (self.cur_idx + 1) % self.max_size


    def __getitem__(self, idx):
        try:
            return self.get(idx)
        except TypeError:
            if idx.start is None:
                start = 0
            else:
                start = idx.start

            if idx.stop is None:
                stop = self.max_size - 1
            else:
                stop = idx.stop

            if idx.step is None:
                step = 1
            else:
                step = idx.step

            return [self[i] for i in range(start, stop, step)]


    def __len__(self):
        if self.is_full:
            return self.max_size

        return self.cur_idx


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    def __enter__(self):
        return self


    def __exit__(self, ty, value, tb):
        self.close()


    @property
    def states(self):
        return self.data_file['states']


    @property
    def actions(self):
        return self.data_file['actions']


    @property
    def rewards(self):
        return self.data_file['rewards']


    @property
    def next_states(self):
        return self.data_file['next_states']


    @property
    def continues(self):
        return self.data_file['continues']


    @property
    def losses(self):
        return self.data_file['losses']


    @property
    def cur_idx(self):
        return self._cur_idx


    @cur_idx.setter
    def cur_idx(self, idx):
        self._cur_idx = idx
        self.data_file.attrs['cur_idx'] = self._cur_idx


    @property
    def is_full(self):
        return self._is_full


    @is_full.setter
    def is_full(self, val):
        self._is_full = val
        self.data_file.attrs['is_full'] = self._is_full


    @property
    def cache_size(self):
        return len(self.cache)
        

#REPLAY SAMPLER PRIORITY
class ReplaySamplerPriority:
    MAX_DUPLICATE_RETRIES = 100


    def __init__(self, replay_memory):
        self.replay_memory = replay_memory
        self.sum_tree = SumTree(self.replay_memory.max_size, dtype='uint32')
        self.add_losses()


    def append(self, state, action, reward, next_state, cont, loss):
        self.sum_tree.add(loss, self.replay_memory.cur_idx)
        self.replay_memory.append(state, action, reward, next_state, cont, loss)


    def sample_memories(self, target, batch_size=32, priorities=None, tree_idxes=None, skip_duplicates=True):
        dup_indexes = {}
        dup_count = 0
        size = self.sum_tree.total / batch_size

        for i in range(batch_size):
            s = random.random() * size + i * size

            t_idx, d_idx, score, memory_idx = self.sum_tree.get_with_info(s)

            self.replay_memory.copy(memory_idx, target, i)

            if priorities is not None:
                priorities[i] = score

            if tree_idxes is not None:
                tree_idxes[i] = t_idx


    def update_sum_tree(self, tree_idxes, losses):
        for t_idx, loss in zip(tree_idxes, losses):
            self.sum_tree.update_score(t_idx, loss)
            memory_idx = self.sum_tree.get_data(t_idx)
            self.replay_memory.set(memory_idx, loss=loss)


    def close(self):
        self.replay_memory.close()


    def add_losses(self):
        for i in range(len(self.replay_memory)):
            self.sum_tree.add(self.replay_memory.losses[i], i)


    def get_min(self):
        return self.sum_tree.get_min()


    def get_max(self):
        return self.sum_tree.get_max()


    def get_average(self):
        return self.sum_tree.get_average()



    def __getitem__(self, idx):
        return self.replay_memory[idx]


    def __len__(self):
        return len(self.replay_memory)


    @property
    def cache_size(self):
        return len(self.replay_memory.cache)


    @property
    def total(self):
        return self.sum_tree.total
