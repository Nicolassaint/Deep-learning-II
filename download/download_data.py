import os
import urllib.request
from torchvision import datasets

def download_file(url, file_path):
    """
    Download a file from the given URL and save it to the specified file path.
    Check if the data directory exists in the project directory, create it if it doesn't,
    and check if the file already exists before downloading.

    Parameters:
        url (str): The URL of the file to download.
        file_path (str): The file path where the downloaded file will be saved.
    """
    # Check if the data directory exists in the project directory, create it if it doesn't
    project_dir = 'Projet'
    data_dir = os.path.join(project_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if the file already exists
    if not os.path.exists(file_path):
        # If the file doesn't exist, download it
        urllib.request.urlretrieve(url, file_path)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

def download_mnist(data_dir='./data/'):
    """
    Download the MNIST Dataset and save it to the specified data directory.

    Parameters:
        data_dir (str): The directory where the MNIST Dataset will be saved.
    """
    # Download the MNIST training set
    mnist_trainset = datasets.MNIST(root=data_dir, train=True, download=True)
    print("MNIST training set downloaded successfully.")

    # Download the MNIST test set
    mnist_testset = datasets.MNIST(root=data_dir, train=False, download=True)
    print("MNIST test set downloaded successfully.")

def main():
    url = 'http://www.cs.nyu.edu/~roweis/data/binaryalphadigs.mat'
    project_dir = 'Projet'
    file_path = os.path.join(project_dir, 'data', 'binaryalphadigs.mat')
    download_file(url, file_path)

    # Download the MNIST Dataset
    download_mnist(data_dir=os.path.join(project_dir, 'data'))

if __name__ == "__main__":
    main()
