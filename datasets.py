"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import string 
import glob
import torchvision
import random
import pickle
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import trange
import logging
def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)



class FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, transform=None,noise_ratio=0):
        super().__init__()
        self.split = 'original-split'
        self.noise_ratio = noise_ratio
        self.path = path
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform
        if not os.path.exists(os.path.join(self.path, 'fashion_iq_data.json')):
            self.fashioniq_data = []
            self.train_init_process()
            with open(os.path.join(self.path, 'fashion_iq_data.json'), 'w') as f:
                json.dump(self.fashioniq_data, f)

            self.test_queries_dress, self.test_targets_dress = self.get_test_data('dress')
            self.test_queries_shirt, self.test_targets_shirt = self.get_test_data('shirt')
            self.test_queries_toptee, self.test_targets_toptee = self.get_test_data('toptee')
            save_obj(self.test_queries_dress, os.path.join(self.path, 'test_queries_dress.pkl'))
            save_obj(self.test_targets_dress, os.path.join(self.path, 'test_targets_dress.pkl'))
            save_obj(self.test_queries_shirt, os.path.join(self.path, 'test_queries_shirt.pkl'))
            save_obj(self.test_targets_shirt, os.path.join(self.path, 'test_targets_shirt.pkl'))
            save_obj(self.test_queries_toptee, os.path.join(self.path, 'test_queries_toptee.pkl'))
            save_obj(self.test_targets_toptee, os.path.join(self.path, 'test_targets_toptee.pkl'))

        else:
            with open(os.path.join(self.path, 'fashion_iq_data.json'), 'r') as f:
                self.fashioniq_data = json.load(f) 
            self.test_queries_dress = load_obj(os.path.join(self.path, 'test_queries_dress.pkl'))
            self.test_targets_dress = load_obj(os.path.join(self.path, 'test_targets_dress.pkl'))
            self.test_queries_shirt = load_obj(os.path.join(self.path, 'test_queries_shirt.pkl'))
            self.test_targets_shirt = load_obj(os.path.join(self.path, 'test_targets_shirt.pkl'))
            self.test_queries_toptee = load_obj(os.path.join(self.path, 'test_queries_toptee.pkl'))
            self.test_targets_toptee = load_obj(os.path.join(self.path, 'test_targets_toptee.pkl'))
            
    def shuffle(self):
        logging.info(f'Shuffle data with noise_ratio {self.noise_ratio}.')
        if self.noise_ratio == 0:
            logging.info("无噪声模式")
            return

        num_samples = len(self.fashioniq_data)
        shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))
        par_p1 = int(len(shuffle_indices) * (1/3))
        par_p2 = int(len(shuffle_indices) * (2/3))
        shuffle_candidate_indices = shuffle_indices[:par_p1]
        shuffle_captions_indices = shuffle_indices[par_p1:par_p2]
        shuffle_target_indices = shuffle_indices[par_p2:]

        noise_candidate = [self.fashioniq_data[i]['candidate'] for i in shuffle_candidate_indices]
        noise_captions = [self.fashioniq_data[i]['captions'] for i in shuffle_captions_indices]
        noise_target = [self.fashioniq_data[i]['target'] for i in shuffle_target_indices]

        random.shuffle(noise_candidate)
        random.shuffle(noise_captions)
        random.shuffle(noise_target)

        for i in shuffle_candidate_indices:
            self.fashioniq_data[i]['candidate'] = noise_candidate.pop()
        for i in shuffle_captions_indices:
            self.fashioniq_data[i]['captions'] = noise_captions.pop()
        for i in shuffle_target_indices:
            self.fashioniq_data[i]['target'] = noise_target.pop()

        logging.info('Shuffle done.')

    def train_init_process(self):
        for name in ['dress', 'shirt', 'toptee']:
            with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(name, 'train')), 'r') as f:
                ref_captions = json.load(f)
            with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f:
                correction_dict = json.load(f)
            for triplets in ref_captions:
                ref_id = triplets['candidate']
                tag_id = triplets['target']
                cap = self.concat_text(triplets['captions'], correction_dict)
                self.fashioniq_data.append({
                    'target': name + '_' + tag_id,
                    'candidate': name + '_' + ref_id,
                    'captions': cap
                })

    def correct_text(self, text, correction_dict):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])

        return text

    def concat_text(self, captions, correction_dict):
        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
        return text

    def __len__(self):
        return len(self.fashioniq_data)

    def __getitem__(self, idx):
        caption = self.fashioniq_data[idx]
        # mod_str = self.concat_text(caption['captions'])
        mod_str = caption['captions']
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}

        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img

    def get_test_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        return img

    def get_all_texts(self):
        texts = []
        for caption in self.fashioniq_data:
            mod_texts = caption['captions']
            texts.append(mod_texts)
        return texts



    def get_test_data(self, name):       # query

        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:
            images = json.load(f)
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(name, 'val')), 'r') as f:
            ref_captions = json.load(f)
        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f:
            correction_dict = json.load(f)
        test_queries = []
        for idx in range(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = self.concat_text(caption['captions'], correction_dict)
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = images.index(candidate)
            out['source_img_data'] = self.get_test_img(name + '_' + candidate)
            out['target_img_id'] = images.index(target)
            out['target_img_data'] = self.get_test_img(name + '_' + target)
            out['mod'] = {'str': mod_str}

            test_queries.append(out)
        
        test_targets_id = []
        test_targets = []
        if self.split == 'val-split':
            for i in test_queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])

            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'] = self.get_img(name + '_' + images[i])
                test_targets.append(out)
        elif self.split == 'original-split':
            for id, image_name in enumerate(images):
                test_targets_id.append(id)
                out = {}
                out['target_img_id'] = id
                out['target_img_data'] = self.get_img(name + '_' + image_name)      
                test_targets.append(out)
        return test_queries, test_targets


class CIRR(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, case_look=False,noise_ratio=0.2) -> None:
        super(CIRR, self).__init__()
        self.path = path
        self.caption_dir = os.path.join(self.path, 'captions')
        self.split_dir = os.path.join(self.path, 'image_splits')
        self.transform = transform
        self.case_look = case_look
        self.noise_ratio=noise_ratio
        # train data
        with open(os.path.join(self.caption_dir, "cap.rc2.train.json"), 'r') as f:
            self.cirr_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.train.json"), 'r') as f:
            self.train_image_path = json.load(f)
            self.train_image_name = list(self.train_image_path.keys())

        with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
            self.val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            self.val_image_path = json.load(f)
            self.val_image_name = list(self.val_image_path.keys())

        # val data
        if not os.path.exists(os.path.join(self.path, 'cirr_val_queries.pkl')):
            self.val_queries, self.val_targets = self.get_val_queries()
            save_obj(self.val_queries, os.path.join(self.path, 'cirr_val_queries.pkl'))
            save_obj(self.val_targets, os.path.join(self.path, 'cirr_val_targets.pkl'))
        else:
            self.val_queries = load_obj(os.path.join(self.path, 'cirr_val_queries.pkl'))
            self.val_targets = load_obj(os.path.join(self.path, 'cirr_val_targets.pkl'))
        # test data
        if not os.path.exists(os.path.join(self.path, 'cirr_test_queries.pkl')):
            self.test_name_list, self.test_img_data, self.test_queries = self.get_test_queries()
            save_obj(self.test_name_list, os.path.join(self.path, 'cirr_test_name_list.pkl'))
            save_obj(self.test_img_data, os.path.join(self.path, 'cirr_test_img_data.pkl'))
            save_obj(self.test_queries, os.path.join(self.path, 'cirr_test_queries.pkl'))
        else:
            self.test_name_list = load_obj(os.path.join(self.path, 'cirr_test_name_list.pkl'))
            self.test_img_data = load_obj(os.path.join(self.path, 'cirr_test_img_data.pkl'))
            self.test_queries = load_obj(os.path.join(self.path, 'cirr_test_queries.pkl'))

    def shuffle(self):
        logging.info(f'Shuffle data with noise_ratio {self.noise_ratio}.')
        if self.noise_ratio == 0:
            logging.info("无噪声模式")
            return
        num_samples = len(self.cirr_data)
        shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))
        par_p1 = int(len(shuffle_indices) * (1/3))
        par_p2 = int(len(shuffle_indices) * (2/3))
        shuffle_candidate_indices = shuffle_indices[:par_p1]
        shuffle_captions_indices = shuffle_indices[par_p1:par_p2]
        shuffle_target_indices = shuffle_indices[par_p2:]

        noise_candidate = [self.cirr_data[i]['reference'] for i in shuffle_candidate_indices]
        noise_captions = [self.cirr_data[i]['caption'] for i in shuffle_captions_indices]
        noise_target = [self.cirr_data[i]['target_hard'] for i in shuffle_target_indices]

        random.shuffle(noise_candidate)
        random.shuffle(noise_captions)
        random.shuffle(noise_target)

        for i in shuffle_candidate_indices:
            self.cirr_data[i]['reference'] = noise_candidate.pop()
        for i in shuffle_captions_indices:
            self.cirr_data[i]['caption'] = noise_captions.pop()
        for i in shuffle_target_indices:
            self.cirr_data[i]['target_hard'] = noise_target.pop()

        logging.info('Shuffle done.')
        
    def __len__(self):
        return len(self.cirr_data)

    def __getitem__(self, idx):
        caption = self.cirr_data[idx]
        reference_name = caption['reference']
        mod_str = caption['caption']
        target_name = caption['target_hard']
        
        out = {}
        out['source_img_data'] = self.get_img(self.train_image_path[reference_name])
        out['target_img_data'] = self.get_img(self.train_image_path[target_name])
        out['mod'] = {'str':mod_str}
        return out

    def get_img(self, img_path, return_raw=False):
        img_path = os.path.join(self.path, img_path.lstrip('./'))
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
            
        if return_raw:
            transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
            return transform(img)

        if self.transform:
            #img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            img = self.transform(img)
        return img

    def get_val_queries(self):
        with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
            val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            val_image_path = json.load(f)
            val_image_name = list(val_image_path.keys())
        
        test_queries = []
        for idx in range(len(val_data)):
            caption = val_data[idx]
            mod_str = caption['caption']
            reference_name = caption['reference']
            target_name = caption['target_hard']
            subset_names = caption['img_set']['members']
            subset_ids = [val_image_name.index(n) for n in subset_names]

            out = {}
            out['source_img_id'] = val_image_name.index(reference_name)
            out['source_img_data'] = self.get_img(val_image_path[reference_name])
            out['target_img_id'] = val_image_name.index(target_name)
            out['target_img_data'] = self.get_img(val_image_path[target_name])
            out['mod'] = {'str':mod_str}
            out['subset_id'] = subset_ids
            if self.case_look:
                out['raw_src_img_data'] = self.get_img(val_image_path[reference_name], return_raw=True)
                out['raw_tag_img_data'] = self.get_img(val_image_path[target_name], return_raw=True)
            
            test_queries.append(out)

        test_targets = []
        for i in range(len(val_image_name)):
            name = val_image_name[i]
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(val_image_path[name])
            if self.case_look:
                out['raw_tag_img_data'] = self.get_img(val_image_path[name], return_raw=True)
            test_targets.append(out)

        return test_queries, test_targets
    
    def get_test_queries(self):

        with open(os.path.join(self.caption_dir, "cap.rc2.test1.json"), 'r') as f:
            test_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.test1.json"), 'r') as f:
            test_image_path = json.load(f)
            test_image_name = list(test_image_path.keys())

        queries = []
        for i in range(len(test_data)):
            out = {}
            caption = test_data[i]
            out['pairid'] = caption['pairid']
            out['reference_data'] = self.get_img(test_image_path[caption['reference']])
            out['reference_name'] = caption['reference']
            out['mod'] = caption['caption']
            out['subset'] = caption['img_set']['members']
            queries.append(out)

        image_name = []
        image_data = []
        for i in range(len(test_image_name)):
            name = test_image_name[i]
            data = self.get_img(test_image_path[name])
            image_name.append(name)
            image_data.append(data)
        return image_name, image_data, queries

