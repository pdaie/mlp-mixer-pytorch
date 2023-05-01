import gdown
import zipfile
import os


def download_dataset():
    print('[!] Loading flood dataset')
    print('- Downloading')
    id_drive = '18LaNRrrHzmicfSExaBIxKrOBSl_Z39yF'
    prefix = 'https://drive.google.com/uc?/export=download&id='
    output = 'data/dog_cat.zip'
    gdown.download(prefix + id_drive,output=output, quiet=True)
    
    print('- Extracting')
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('data')
        
    os.remove(output)
    print('- Done')
    print()