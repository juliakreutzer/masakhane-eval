from model_loader import MasakhaneModelLoader
import argparse

tsv_file = 'https://raw.githubusercontent.com/juliakreutzer/masakhane-eval/master/models/available_models.tsv'

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Masakhane model loader")
    ap.add_argument("src_lang", type=str, help="Source language.")
    ap.add_argument("trg_lang", type=str, help="Target language.")
    ap.add_argument("--domain", type=str, help="Domain.", default='JW300')
    args = ap.parse_args()
    mloader = MasakhaneModelLoader(tsv_file, src_language=args.src_lang,
                                   domain=args.domain)
    #mloader.load_and_check_all_models()
    if args.trg_lang not in mloader.models:
        print('Sorry, model not available.')
    else:
        model_dir, config, lc = mloader.download_model(
            trg_language=args.trg_lang)
        print('Please find your JoeyNMT Masakhane model in {}.'.format(
            model_dir))
        print('This model is {}lowercased.'.format('' if lc else 'not '))
