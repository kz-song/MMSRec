## install git-lfs
# sudo apt-get install git-lfs
#
## download dataset
# git lfs clone https://huggingface.co/datasets/iejMac/CLIP-WebVid.git

cd ../raw
mv ./CLIP-WebVid/data/train ./train
mv ./CLIP-WebVid/data/val ./eval
rm -rf ./CLIP-WebVid

# unzip tar file and delete tar file
cd ./train
ls ./*.tar > ls.log
for file in $(cat ls.log)
do
  echo "unzip file ${file} to ${file%.*}"
  mkdir ${file%.*}
  tar -xf ${file} -C ${file%.*}
  rm ${file}
done
rm ls.log
cd ../

cd ./eval
ls ./*.tar > ls.log
for file in $(cat ls.log)
do
  echo "unzip file ${file} to ${file%.*}"
  mkdir ${file%.*}
  tar -xf ${file} -C ${file%.*}
  rm ${file}
done
rm ls.log

