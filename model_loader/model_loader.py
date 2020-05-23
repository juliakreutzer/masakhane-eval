import urllib.request
import requests
import os
from joeynmt import helpers
import yaml
from time import sleep


class MasakhaneModelLoader:
    """Load Masakhane models given a table with urls to their parts."""

    def __init__(self, model_parts_url, src_language='en', domain='JW300'):
        models_str = urllib.request.urlopen(model_parts_url).read().decode("utf-8")
        self._model_dir_prefix = 'joeynmt/models/'

        self._src_language = ''
        self.models = self.load_available_models(models_str,
                                                 src_language, domain)

    def load_available_models(self, models_str,
                              src_language='en', domain='JW300'):
        """
        Get list of available models.
        If multiple models: select domain.
        Only select relevant models with correct src language.

        :param models_str: string that contains all model urls
        :param src_language: source language
        :param domain: selected domain
        :return: dictionaries by target language that contains model urls.
        """

        models = {}
        for i, line in enumerate(models_str.split('\n')):
            entries = line.strip().split("\t")
            if i == 0:
                headers = entries
                header_keys = [h.__str__() for h in headers]
                continue
            model = {h: v for h, v in zip(header_keys, entries)}
            if (model['src_language'] != src_language or
                        model['complete'] != 'yes'):
                continue
            if (model['trg_language'] in models.keys() and
                        model['domain'] != domain):
                continue
            models[model['trg_language']] = model
        print('Found {} Masakhane models.'.format(len(models)))
        self._model_dir_prefix += src_language
        self._src_language = src_language
        return models

    def load_and_check_all_models(self):
        """Try to download all model files."""
        completed_models = []
        failed_models = []
        for trg_lang, modelpaths in self.models.items():
            model = self.download_model(trg_lang)
            if model is not None:
                completed_models.append(trg_lang)
            else:
                failed_models.append(trg_lang)
        print('Model loading succeeded for {}/{} models: {}'.format(
            len(completed_models), len(self.models), completed_models))
        print('Model loading failed for {}/{} models: {}'.format(
            len(failed_models), len(self.models), failed_models))

    def download_model(self, trg_language):
        """ Download model for given trg language. """
        model_dir = "{}-{}".format(self._model_dir_prefix, trg_language)
        failed = 0
        print("Downloading model for ", trg_language, "to ", model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print("Directory ", model_dir, " created. ")
        model_files = self.models[trg_language]
        # Download the checkpoint.
        ckpt_path = os.path.join(model_dir, 'model.ckpt')
        failed += self._download(model_files['ckpt'], ckpt_path)
        # Download the vocabularies.
        src_vocab_file = model_files['src_vocab']
        trg_vocab_file = model_files['trg_vocab']
        src_vocab_path = os.path.join(model_dir, 'src_vocab.txt')
        failed += self._download(src_vocab_file, src_vocab_path)
        trg_vocab_path = os.path.join(model_dir, 'trg_vocab.txt')
        failed += self._download(trg_vocab_file, trg_vocab_path)
        # Download the config.
        config_file = model_files['config.yaml']
        config_path = os.path.join(model_dir, 'config_orig.yaml')
        failed += self._download(config_file, config_path)
        # Adjust config.
        config = helpers.load_config(config_path)
        new_config_file = os.path.join(model_dir, 'config.yaml')
        config = self._update_config(config, src_vocab_path, trg_vocab_path,
                                     model_dir, ckpt_path)
        with open(new_config_file, 'w') as cfile:
            yaml.dump(config, cfile)
        # Download BPE codes.
        src_bpe_path = os.path.join(model_dir, 'src.bpe.model')
        trg_bpe_path = os.path.join(model_dir, 'trg.bpe.model')
        failed += self._download(model_files['src_bpe'], src_bpe_path)
        failed += self._download(model_files['trg_bpe'], trg_bpe_path)
        if failed >= 1:
            print('Download of model for {}-{} FAILED.'.format(
                self._src_language, trg_language))
            return None
        else:
            print('Downloaded model for {}-{} successfully.'.format(
                self._src_language, trg_language))
            return model_dir, config, self._is_lc(src_vocab_path)

    def _update_config(self, config, new_src_vocab_path, new_trg_vocab_path,
                       new_model_dir, new_ckpt_path):
        """ Overwrite the settings in the given config.
        :param config:
        :param new_src_vocab_path:
        :param new_trg_vocab_path:
        :param new_model_dir:
        :param new_ckpt_path:
        :return:
        """
        assert config, 'Configuration is empty!'
        assert 'data' in config.keys(), 'Configuration is missing data section!'
        assert 'model' in config.keys(), 'Configuration is missing model section!'
        assert 'training' in config.keys(), 'Configuration is missing training section!'
        config['data']['src_vocab'] = new_src_vocab_path
        if config['model'].get('tied_embeddings', False):
            config['data']['trg_vocab'] = new_src_vocab_path
        else:
            config['data']['trg_vocab'] = new_trg_vocab_path
        config['training']['model_dir'] = new_model_dir
        config['training']['load_model'] = new_ckpt_path
        return config

    def _is_lc(self, src_vocab_path):
        """ Infer whether the model is built on lowercased data."""
        lc = True
        with open(src_vocab_path, 'r') as ofile:
            for line in ofile:
                if line != line.lower():
                    lc = False
                    break
        return lc

    def _download_gdrive_file(self, file_id, destination):
        """Download a file from Google Drive and store in local file."""
        #download_link = 'https://drive.google.com/uc?id={}'.format(file_id)
        url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        chunk_size = 32123
        try:
            response = session.get(url, params={'id': file_id}, stream=True)
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
            if token:
                params = {'id': id, 'confirm': token}
                response = session.get(url, params=params, stream=True)
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
        except:
            print("Couldn't download Gdrive file with id {}.".format(file_id))
            return 1
        return 0

    def _download_github_file(self, github_raw_path, destination):
        """Download a file from GitHub."""
        sleep(1)
        try:
            urllib.request.urlretrieve(github_raw_path, destination)
        except:
            print("Couldn't download {}.".format(github_raw_path))
            return 1
        return 0

    def _download(self, url, destination):
        """Download file from Github or Googledrive."""
        failed = 1
        sleep(1)
        try:
            if 'drive.google.com' in url:
                if url.startswith('https://drive.google.com/file'):
                    file_id = url.split("/")[-1]
                elif url.startswith('https://drive.google.com/open?'):
                    file_id = url.split('id=')[-1]
                failed = self._download_gdrive_file(file_id, destination)
            else:
                failed = self._download_github_file(url, destination)
        except:
            print("Download failed, didn't recognize url {}.".format(url))
        return failed