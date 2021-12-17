# Deprecated not working currently 

import gdown

# https://drive.google.com/file/d/1-V2wbyOyWXWfJeUHNXRnGNhCim6QBV3_/view?usp=sharing
url = 'https://drive.google.com/uc?id=1-V2wbyOyWXWfJeUHNXRnGNhCim6QBV3_'
output = './resources/psp/model_ir_se50.pth'  
gdown.download(url, output, quiet=False)