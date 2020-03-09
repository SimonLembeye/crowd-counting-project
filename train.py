from models import vgg19, CSRNet
from loaders import loading_data_GT,  loading_data_Bayes

csr = CSRNet()
print(csr)


_, _, _ = loading_data_Bayes()
