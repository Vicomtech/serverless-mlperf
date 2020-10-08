#!/usr/bin/env python
import os
import zipfile
import os
import shutil
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

if __name__ == '__main__':
    os.chdir('handler_utils')
    zipf = zipfile.ZipFile('handler_utils.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('python', zipf)
    zipf.close()
    original = 'handler_utils.zip'
    target = '../'

    shutil.move(original, target)