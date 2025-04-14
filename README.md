# DiCARN-DNase: Improving Cell-to-Cell Hi-C Resolution Using Dilated Cascading ResNet and DNase-seq Chromatin Accessibility Data
Bioinformatics Lab, University of Colorado, Colorado Springs

### Developer:
Samuel Olowofila

Department of Computer Science

University of Colorado Colorado Springs

Email: solowofi@uccs.edu


### Contact:

Oluwatosin Oluwadare, PhD

Department of Computer Science

University of Colorado Colorado Springs

Email: ooluwada@uccs.edu

## Build Instructions:
To ensure seamless explorations of DiCARN-DNase, this project is containarized on the Docker platform.
Follow the steps below to install and build:
1. Clone this repository using the git command provided: `git clone https://github.com/OluwadareLab/DiCARN_DNase.git && cd DiCARN_DNase`
2. Pull the DiCARN-DNase docker image from the Docker Hub using the command `docker pull oluwadarelab/dicarn_dnase:latest`. Verify the successful download of the image using the command `docker image ls`
3. Run an instance of the image while ensuring that the current working directory is mounted on the container. Try `docker run --rm -it --gpus all --name dicarn_dnase -v ${PWD}:${PWD} oluwadarelab/dicarn_dnase`
4. cd to your home directory.


## DiCARN-DNase Dependencies
Below is a list of recommended dependency versions for running this project:
- Python 3.8
- Cuda 12.2
- Pytorch 1.10
- Matplotlib 3.5
- Numpy 1.22
- Scikit-learn
- Scipy 1.8
- tqdm 4.64
- Pandas 2.2

## Modes of Running Software
1. You can opt for a quick start using our already processed data. This is made available on [Zenodo](https://zenodo.org/records/15198848). Subsequently, follow the steps under the "Prediction with Analysis" section to predict and analyze.
2. You can follow the data pre-processing outline below to manually pre-process your data.


## Data Pre-processing
Access the GSE62525 GEO entry for Hi-C data from (Rao et al., 2014) [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525). In our work, we made use of the [GM12878](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Fprimary%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz) primary intrachromosomal, [K562](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FK562%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz) intrachromosomal, [HMEC](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FHMEC%5Fintrachromosomal%5Fcontact%5Fmatrices.tar.gz) intrachromosomal, and [NHEK](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FNHEK%5Fintrachromosomal%5Fcontact%5Fmatrices.tar.gz) intrachromosomal matrices.

 - Set your project's root directory by defining it as a string in `Data/Arg_Parser.py`. For instance, we have root_dir = './Data'.
 - Create a folder named raw within the root directory to hold your unprocessed datasets. Use this command: `mkdir $root_dir/raw`.
 - Download and extract your datasets into the `$root_dir/raw` directory. For GM12878 data, an appropriately named folder will be created containing the contact matrices for every chromosome and available resolution. Refer to the README for additional guidance.

To generate .npz formatted datasets for DiCARN, follow the outlined steps:

### 1. Processing Raw Data
This step will generate a new directory $root_dir/mat/<cell_line_name> containing all chrN_[HR].npz files.

`python3 Read_Data.py -c GM12878`

Mandatory argument:

`-c`: Specify the name of the folder where you saved the unzipped cell line data under $root_dir/raw/<cell_line_name>. In this example, <cell_line_name> is GM12878.
Optional arguments:

`-hr`: Resolution setting. Options include 5kb, 10kb, 25kb, 50kb, 100kb, 250kb, 500kb, and 1mb. Default is set to 10kb.

`-q`: Map quality setting. Choices are MAPQGE30 or MAPQG0. Default is MAPQGE30.

`-n`: Normalization approach. Choose from KRnorm, SQRTVCnorm, or VCnorm. Default is KRnorm.


### 2. Downsampling Data Randomly
This step stores downsampled HR data in $root_dir/mat/<cell_line_name> as chrN_[LR].npz.

`python Downsample.py -hr 10kb -lr 40kb -r 16 -c GM12878`

Arguments:

`-hr`: Resolution from the previous step. Default is 10kb.

`-lr`: Specifies the resolution for [LR] in chrN_[LR].npz. Default is 40kb.

`-r`: Downsampling factor. Default value is 16.

`-c`: Cell line identifier.

### 3. Creating Training, Validation, and Testing Datasets
 - Specify chromosomes for each dataset category in `./Arg_Parser.py` within the `set_dict` dictionary.

 - The example below generates a file in `$root_dir/Data with the name dicarn_10kb40kb_c40_s40_b201_nonpool_train.npz`:

`python Generate.py -hr 10kb -lr 40kb -lrc 100 -s train -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878`

Arguments:

`-hr`: High resolution for chrN_[HR].npz, which serves as the target data for training. Default is 10kb.

`-lr`: Low resolution for chrN_[LR].npz used as input data during training. Default is 40kb.

`-lrc`: Sets the minimum value in the LR matrix. Default is 100.

`-s`: Type of dataset to generate. Options are train, valid, GM12878_test, K562_test, NHEK_test and HMEC_test. Default is train.

`-chunk`: Defines the submatrix size (nxn). Default is 40.

`-stride`: Set equal to -chunk. Default is 40.

`-bound`: Maximum genomic distance limit. Default is 201.

`-scale`: Determines whether input submatrices should be pooled. Default is 1.

`-c`: Specify the cell line name again.

Note: To proceed with training, ensure that both training and validation files are present in $root_dir/data. Adjust the -s option to produce validation and additional required datasets.
### Congratulations! Your datasets are now ready.


## Training
To initiate training, run:
`python DiCARN_Train.py`

Running this script will generate .pytorch checkpoint files containing the model’s trained weights. 

During the validation process, if a new peak SSIM score is achieved, the corresponding epoch’s weights will be stored as `bestV`. This may result in multiple `bestV` checkpoint files throughout training. Upon completing all training epochs, a finalg checkpoint file will be saved. We used the `finalV` files for predictions in our study.

## Prediction with Analysis
Pretrained weights for DiCARN and other comparison models are provided. Alternatively, you can use weights from your own trained models. For quick predictions, execute the following commands:

`python Predict_DiCARN.py -m DiCARN -lr 40kb -ckpt root_dir/checkpoints/<weights_filename>.pytorch -f dicarn_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test.npz -c GM12878_DiCARN`

Prediction Command Arguments:

`-m`: Specifies the model for prediction. Choose from DiCARN, DFHiC, or HiCSR.

`-lr`: Low-resolution input to be enhanced. Default is 40kb.

`-ckpt`: Path to the checkpoint file, either from our Pretrained_weights or from your $root_dir/checkpoints.

`-f`: Filename of the low-resolution dataset to enhance. It must be located in $root_dir/data.

`-c`: The cell line identifier. Example: `dicarn_10kb40kb_c40_s40_b201_nonpool_GM12878_test.npz`.

Be sure to adjust all arguments accordingly.

## Accessing Predicted Data
The predictions are saved in `.npz` files that store NumPy arrays using specific keys. Your predicted high-resolution (HR) contact map can be found under the dicarn key, while the compact key holds indices for non-zero entries in the contact map.

To retrieve the predicted HR matrix, use the following line in a Python script:

`hic_matrix = np.load("path/to/file.npz", allow_pickle=True)['dicarn']`

## DNase-seq Data Processing
The derivation of interaction frequency data using DNase-seq data is done using the `DNase_imputation.R` script. 

### Directory Structure
The example directory structure below ensures the script can locate each file at the expected paths for smooth execution. 

```plaintext
~/DNase_Data/HMEC/
├── Src/
│   ├── chr1/
│   │   ├── chr1_10kb.RAWobserved
│   │   ├── chr1_10kb.KRnorm
│   │   └── chr1.bed
│   ├── chr3/
│   │   ├── chr3_10kb.RAWobserved
│   │   ├── chr3_10kb.KRnorm
│   │   └── chr3.bed
│   └── (similar structure for each chromosome in `chromosomes`)
└── exponential_linear_model_-0.4.Rdata
```

### DNase-seq preprocessing exhibition
Simply run the command below to preprocess the DNase-seq data:
`Rscript DNase_imputation.R`

The output interaction frequency file in `.tsv` format can then be adopted for various uses. A useful example as relating to the DiCARN-DNase project is converting the file to `.coo` format and running it through the Hi-C data preprocessing pipeline earlier discussed (Examples are made available in the `DNase` subdirectory of the `dicarn_project_data` availabe on our [Zenodo repository](https://zenodo.org/records/14009929). This data is then concatenated with the GM12878 data for training our DiCARN-DNase per cell line. 
